import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from backend.config import get_llm, get_provider
from backend.vectorstore import get_vectorstore


class GraphState(TypedDict, total=False):
    original_query: str
    rewritten_query: str
    image_data: Optional[str]
    image_mime_type: Optional[str]
    image_analysis: Optional[str]
    conversation_history: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    rerank_scores: Dict[str, float]
    relevance_verdict: str
    relevance_feedback: str
    retry_count: int
    final_response: str
    pipeline_route: List[str]
    response_mode: str
    session_id: str
    _requires_rag: bool
    _raw_orig_docs: List[Any]
    _raw_rewritten_docs: List[Any]


class DamageAssessment(BaseModel):
    damage_location: str = Field(description="Primary visible location of the vehicle damage.")
    affected_parts: List[str] = Field(description="Visible parts that appear damaged.")
    apparent_severity: str = Field(description="One of: minor, moderate, severe, or unclear.")
    drivability_indicators: List[str] = Field(
        description="Visible signs about drivability such as leaks, wheel alignment, or deployed airbags."
    )
    notes: str = Field(description="Short factual explanation of what can be seen in the image.")


class OrchestratorVerdict(BaseModel):
    requires_rag: bool = Field(
        description="True when document retrieval is needed to answer accurately from policy, claims, or coverage materials."
    )
    reason: str = Field(description="Short explanation for the routing decision.")


class RelevanceVerdict(BaseModel):
    is_relevant_and_sufficient: bool = Field(
        description="True if the retrieved documents are relevant and sufficient to answer the original and rewritten queries."
    )
    feedback_reason: str = Field(
        description="If false, explain what information is missing or why the documents are insufficient."
    )


def _extract_message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _invoke_structured_output(llm: Any, schema: type[BaseModel], prompt: str) -> BaseModel:
    try:
        return llm.with_structured_output(schema).invoke(prompt)
    except Exception:
        json_hint = schema.model_json_schema()
        fallback = llm.invoke(
            f"{prompt}\n\nRespond with JSON only that matches this schema:\n{json.dumps(json_hint)}"
        )
        text = _extract_message_text(fallback)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse structured output for {schema.__name__}. Raw output: {text}")
        return schema.model_validate_json(match.group(0))


def _format_history(history: List[Dict[str, Any]], max_turns: int = 8) -> str:
    if not history:
        return "None"

    formatted: List[str] = []
    for item in history[-max_turns:]:
        role = str(item.get("role", "user")).upper()
        content = str(item.get("content", "")).strip()
        if content:
            formatted.append(f"{role}: {content}")
        image_analysis = item.get("image_analysis")
        if image_analysis:
            formatted.append(f"{role}_IMAGE_CONTEXT: {image_analysis}")
    return "\n".join(formatted) if formatted else "None"


def _serialize_image_analysis(assessment: DamageAssessment) -> str:
    parts = ", ".join(assessment.affected_parts) if assessment.affected_parts else "unspecified parts"
    drivability = "; ".join(assessment.drivability_indicators) if assessment.drivability_indicators else "none noted"
    return (
        f"Location: {assessment.damage_location}. "
        f"Affected parts: {parts}. "
        f"Apparent severity: {assessment.apparent_severity}. "
        f"Drivability indicators: {drivability}. "
        f"Notes: {assessment.notes}"
    )


def _build_vision_message(prompt: str, image_base64: str, mime_type: Optional[str]) -> HumanMessage:
    media_type = mime_type or "image/jpeg"
    provider = get_provider()

    if provider == "anthropic":
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64,
                    },
                },
            ]
        )

    if provider in {"openai", "google", "groq"}:
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
                },
            ]
        )

    raise ValueError(
        "The configured provider does not support local image uploads for the vision step."
    )


def analyze_image(state: GraphState) -> Dict[str, Any]:
    route = list(state.get("pipeline_route", []))
    image_data = state.get("image_data")
    mime_type = state.get("image_mime_type")

    if not image_data:
        route.append("Vision Analysis: skipped (no image uploaded)")
        return {"image_analysis": None, "pipeline_route": route}

    llm = get_llm(vision=True, temperature=0.0)
    prompt = (
        "You are assessing a vehicle damage photo for an auto-insurance triage workflow.\n"
        "Describe only what is visible. Do not guess hidden structural damage.\n"
        "Return structured fields for location, affected parts, severity, drivability indicators, and notes."
    )
    message = _build_vision_message(prompt, image_data, mime_type)
    try:
        assessment = llm.with_structured_output(DamageAssessment).invoke([message])
    except Exception:
        fallback = llm.invoke(
            [
                _build_vision_message(
                    f"{prompt}\nRespond with JSON only for these fields: damage_location, affected_parts, apparent_severity, drivability_indicators, notes.",
                    image_data,
                    mime_type,
                )
            ]
        )
        text = _extract_message_text(fallback)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse image analysis output. Raw output: {text}")
        assessment = DamageAssessment.model_validate_json(match.group(0))
    route.append("Vision Analysis: completed")
    return {
        "image_analysis": _serialize_image_analysis(assessment),
        "pipeline_route": route,
    }


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    llm = get_llm(vision=False, temperature=0.0)
    route = list(state.get("pipeline_route", []))

    prompt = (
        "You are a query rewriting specialist for an auto-insurance RAG system.\n"
        "Rewrite the user's request into a concise retrieval query that preserves intent and prior context.\n"
        "If a damage-photo analysis exists, fuse it into the retrieval query.\n"
        "Target terms such as collision/comprehensive, deductible, documentation, total-loss handling, exclusions, repair, and claim steps when relevant.\n"
        "Return only the rewritten query.\n\n"
        f"Conversation History:\n{_format_history(state.get('conversation_history', []))}\n\n"
        f"Image Analysis:\n{state.get('image_analysis') or 'None'}\n\n"
        f"Original User Query:\n{state.get('original_query', '').strip()}"
    )

    rewritten_query = _extract_message_text(llm.invoke(prompt)) or state.get("original_query", "").strip()
    route.append(f"Query Rewrite: {rewritten_query}")
    return {"rewritten_query": rewritten_query, "pipeline_route": route}


def orchestrate(state: GraphState) -> Dict[str, Any]:
    llm = get_llm(vision=False, temperature=0.0)
    route = list(state.get("pipeline_route", []))

    prompt = (
        "You are the routing orchestrator for an auto-insurance assistant.\n"
        "Decide whether this request requires retrieval from policy/claims documents.\n"
        "Use requires_rag=false only for greetings, conversational filler, or questions answerable from chat context alone.\n\n"
        f"Original Query: {state.get('original_query', '')}\n"
        f"Rewritten Query: {state.get('rewritten_query', '')}\n"
        f"Conversation History:\n{_format_history(state.get('conversation_history', []), max_turns=4)}"
    )

    verdict = _invoke_structured_output(llm, OrchestratorVerdict, prompt)
    route.append("Route Decision: RAG" if verdict.requires_rag else "Route Decision: Direct")

    return {
        "_requires_rag": verdict.requires_rag,
        "relevance_verdict": "PENDING" if verdict.requires_rag else "DIRECT",
        "pipeline_route": route,
        "retry_count": state.get("retry_count", 0),
    }


def retrieve_docs(state: GraphState) -> Dict[str, Any]:
    route = list(state.get("pipeline_route", []))
    retry_count = state.get("retry_count", 0)
    label = f"Retrieve Documents (retry {retry_count})" if retry_count else "Retrieve Documents"
    route.append(label)

    try:
        db = get_vectorstore(create_if_missing=False)
        docs_orig = db.similarity_search(state.get("original_query", ""), k=6)
        docs_rewritten = db.similarity_search(state.get("rewritten_query", ""), k=6)
    except Exception:
        docs_orig = []
        docs_rewritten = []
        route.append("Retrieve Documents: vector store unavailable or empty")

    return {
        "_raw_orig_docs": docs_orig,
        "_raw_rewritten_docs": docs_rewritten,
        "pipeline_route": route,
    }


def rerank_docs(state: GraphState) -> Dict[str, Any]:
    route = list(state.get("pipeline_route", []))
    route.append("Re-rank Documents: Reciprocal Rank Fusion")

    docs_orig = state.get("_raw_orig_docs", []) or []
    docs_rewritten = state.get("_raw_rewritten_docs", []) or []
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    def process_docs(docs: List[Any]) -> None:
        for rank, doc in enumerate(docs, start=1):
            key = f"{doc.metadata.get('source_url', doc.metadata.get('source', 'unknown'))}::{hash(doc.page_content)}"
            scores[key] = scores.get(key, 0.0) + (1.0 / (60 + rank))
            doc_map[key] = doc

    process_docs(docs_orig)
    process_docs(docs_rewritten)

    top_keys = [key for key, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:6]]
    retrieved_docs: List[Dict[str, Any]] = []
    rerank_scores: Dict[str, float] = {}

    for key in top_keys:
        doc = doc_map[key]
        score = scores[key]
        retrieved_docs.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "source_title": doc.metadata.get("source_title", doc.metadata.get("source", "unknown")),
                "source_url": doc.metadata.get("source_url"),
                "rrf_score": score,
            }
        )
        rerank_scores[key] = score

    return {
        "retrieved_docs": retrieved_docs,
        "rerank_scores": rerank_scores,
        "pipeline_route": route,
    }


def evaluate_relevance(state: GraphState) -> Dict[str, Any]:
    route = list(state.get("pipeline_route", []))
    llm = get_llm(vision=False, temperature=0.0)
    docs = state.get("retrieved_docs", [])

    if not docs:
        route.append("Evaluate Relevance: NO")
        return {
            "relevance_verdict": "NO",
            "relevance_feedback": "No documents were retrieved from the knowledge base.",
            "pipeline_route": route,
        }

    docs_text = "\n\n".join(
        [
            (
                f"[Document {index}] Title: {doc['source_title']}\n"
                f"URL: {doc.get('source_url') or 'n/a'}\n"
                f"Excerpt:\n{doc['content']}"
            )
            for index, doc in enumerate(docs, start=1)
        ]
    )

    prompt = (
        "You are checking whether retrieved insurance documents are enough to answer the user accurately.\n"
        "Evaluate both the original query and the rewritten retrieval query.\n"
        "Mark false if the documents are off-topic, incomplete, or do not clearly support the answer.\n\n"
        f"Original Query: {state.get('original_query', '')}\n"
        f"Rewritten Query: {state.get('rewritten_query', '')}\n\n"
        f"Retrieved Documents:\n{docs_text}"
    )

    verdict = _invoke_structured_output(llm, RelevanceVerdict, prompt)
    relevance_verdict = "YES" if verdict.is_relevant_and_sufficient else "NO"
    route.append(f"Evaluate Relevance: {relevance_verdict}")

    return {
        "relevance_verdict": relevance_verdict,
        "relevance_feedback": verdict.feedback_reason,
        "pipeline_route": route,
    }


def retry_rewrite_query(state: GraphState) -> Dict[str, Any]:
    llm = get_llm(vision=False, temperature=0.0)
    retry_count = state.get("retry_count", 0) + 1
    route = list(state.get("pipeline_route", []))

    prompt = (
        "You are refining a failed retrieval query for an auto-insurance RAG system.\n"
        "Use the evaluator feedback to produce a better retrieval query.\n"
        "Return only the improved query.\n\n"
        f"Original Query: {state.get('original_query', '')}\n"
        f"Previous Rewritten Query: {state.get('rewritten_query', '')}\n"
        f"Evaluator Feedback: {state.get('relevance_feedback', '')}\n"
        f"Conversation History:\n{_format_history(state.get('conversation_history', []), max_turns=4)}"
    )

    rewritten_query = _extract_message_text(llm.invoke(prompt)) or state.get("rewritten_query", "")
    route.append(f"Retry Query Rewrite #{retry_count}: {rewritten_query}")
    return {
        "rewritten_query": rewritten_query,
        "retry_count": retry_count,
        "pipeline_route": route,
    }


def generate_grounded_response(state: GraphState) -> Dict[str, Any]:
    llm = get_llm(vision=False, temperature=0.1)
    route = list(state.get("pipeline_route", []))
    docs_text = "\n\n".join(
        [
            (
                f"[Document {index}] Title: {doc['source_title']}\n"
                f"URL: {doc.get('source_url') or 'n/a'}\n"
                f"Excerpt:\n{doc['content']}"
            )
            for index, doc in enumerate(state.get("retrieved_docs", []), start=1)
        ]
    )

    prompt = (
        "You are ClaimSight AI, an auto-insurance claims triage assistant.\n"
        "Answer using only the retrieved documents plus prior conversation context.\n"
        "If something is not supported by the documents, say so clearly.\n"
        "Cite the document title in-line when you make a grounded claim.\n"
        "Be concise, practical, and specific about next steps.\n"
        "End with the exact disclaimer line provided below.\n\n"
        f"Conversation History:\n{_format_history(state.get('conversation_history', []))}\n\n"
        f"Image Analysis:\n{state.get('image_analysis') or 'None'}\n\n"
        f"Original Query:\n{state.get('original_query', '')}\n\n"
        f"Rewritten Query:\n{state.get('rewritten_query', '')}\n\n"
        f"Retrieved Documents:\n{docs_text}\n\n"
        "Disclaimer line:\n"
        "Disclaimer: This is triage guidance only, not a final coverage decision. A licensed adjuster and your active policy documents control the actual outcome."
    )

    final_response = _extract_message_text(llm.invoke(prompt))
    if "Disclaimer:" not in final_response:
        final_response = (
            f"{final_response}\n\n"
            "Disclaimer: This is triage guidance only, not a final coverage decision. "
            "A licensed adjuster and your active policy documents control the actual outcome."
        )

    route.append("Generate Grounded Response")
    return {
        "final_response": final_response,
        "response_mode": "rag",
        "pipeline_route": route,
    }


def generate_direct_response(state: GraphState) -> Dict[str, Any]:
    llm = get_llm(vision=False, temperature=0.4)
    route = list(state.get("pipeline_route", []))

    prompt = (
        "You are ClaimSight AI.\n"
        "Respond naturally to the user's message using only conversation context.\n"
        "Do not invent policy details or cite documents when the request does not need retrieval.\n\n"
        f"Conversation History:\n{_format_history(state.get('conversation_history', []))}\n\n"
        f"User Query:\n{state.get('original_query', '')}"
    )

    final_response = _extract_message_text(llm.invoke(prompt))
    route.append("Generate Direct Response")
    return {
        "final_response": final_response,
        "response_mode": "direct",
        "pipeline_route": route,
    }


def generate_safe_response(state: GraphState) -> Dict[str, Any]:
    route = list(state.get("pipeline_route", []))
    route.append("Generate Safe Response")
    feedback = state.get("relevance_feedback") or "The retrieved documents did not cover the missing detail."

    final_response = (
        "I could not find enough grounded information in the current claims and policy corpus to answer that reliably.\n\n"
        f"Why the search fell short: {feedback}\n\n"
        "To help me answer better, please clarify one or more of these:\n"
        "- the state tied to the policy or claim\n"
        "- whether the loss is collision, comprehensive, theft, vandalism, or total-loss related\n"
        "- any deductible, rental, police-report, or repair-parts question you want answered\n"
        "- a damage photo if the question depends on what is visible\n\n"
        "Disclaimer: This is triage guidance only, not a final coverage decision. A licensed adjuster and your active policy documents control the actual outcome."
    )

    return {
        "final_response": final_response,
        "response_mode": "safe",
        "pipeline_route": route,
    }


def route_after_orchestrate(state: GraphState) -> str:
    return "retrieve" if state.get("_requires_rag", True) else "direct_response"


def route_after_evaluate(state: GraphState) -> str:
    if state.get("relevance_verdict") == "YES":
        return "rag_response"
    if state.get("retry_count", 0) < 2:
        return "retry_rewrite"
    return "safe_response"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze_image", analyze_image)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("orchestrate", orchestrate)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("rerank", rerank_docs)
    workflow.add_node("evaluate", evaluate_relevance)
    workflow.add_node("retry_rewrite", retry_rewrite_query)
    workflow.add_node("rag_response", generate_grounded_response)
    workflow.add_node("direct_response", generate_direct_response)
    workflow.add_node("safe_response", generate_safe_response)

    workflow.add_edge(START, "analyze_image")
    workflow.add_edge("analyze_image", "rewrite_query")
    workflow.add_edge("rewrite_query", "orchestrate")
    workflow.add_conditional_edges(
        "orchestrate",
        route_after_orchestrate,
        {
            "retrieve": "retrieve",
            "direct_response": "direct_response",
        },
    )
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "rag_response": "rag_response",
            "retry_rewrite": "retry_rewrite",
            "safe_response": "safe_response",
        },
    )
    workflow.add_edge("retry_rewrite", "retrieve")
    workflow.add_edge("rag_response", END)
    workflow.add_edge("direct_response", END)
    workflow.add_edge("safe_response", END)
    return workflow.compile()


claimsight_agent = build_graph()
