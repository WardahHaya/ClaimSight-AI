import argparse
import base64
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db
from backend.graph import claimsight_agent
from backend.vectorstore import get_vectorstore


def _encode_image(path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not path:
        return None, None

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    mime_type = "image/jpeg"
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"

    return base64.b64encode(file_path.read_bytes()).decode("utf-8"), mime_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClaimSight LangGraph pipeline without the API layer.")
    parser.add_argument("query", help="User query to send through the graph.")
    parser.add_argument("--session-id", default="cli-test-session")
    parser.add_argument("--image", help="Optional path to a vehicle-damage image.")
    args = parser.parse_args()

    init_db()
    try:
        vectorstore = get_vectorstore(create_if_missing=False)
        print(f"Vector store ready. Collection count: {vectorstore._collection.count()}")
    except Exception as exc:
        print(f"Vector store unavailable: {exc}")

    image_data, image_mime_type = _encode_image(args.image)
    result = claimsight_agent.invoke(
        {
            "original_query": args.query,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "conversation_history": [],
            "session_id": args.session_id,
            "retry_count": 0,
            "pipeline_route": ["Load Conversation History"],
        }
    )

    print("\nPipeline route:")
    for step in result.get("pipeline_route", []):
        print(f"- {step}")

    if result.get("image_analysis"):
        print("\nImage analysis:")
        print(result["image_analysis"])

    print("\nFinal response:")
    print(result.get("final_response", "No response generated."))


if __name__ == "__main__":
    main()
