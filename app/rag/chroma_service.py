"""
ChromaDB Service — persistent vector store for hotel FAQs and policies.

Stores: Policies, Amenities, FAQs, Hotel information.
Does NOT store: Booking data, pricing, room availability.

Uses sentence-transformers for embeddings (all-MiniLM-L6-v2).
"""

import logging
from pathlib import Path

import chromadb

from app.config.settings import get_settings
from app.db.mongodb import knowledge_base_collection, hotel_kb_values_collection

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "hotel_knowledge"


def _reset_chroma_store() -> None:
    """Drop in-memory handles and remove a corrupted on-disk ChromaDB store."""
    global _client, _collection
    _client = None
    _collection = None
    settings = get_settings()
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    if not persist_dir.exists():
        return
    for pattern in ("*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm"):
        for path in persist_dir.glob(pattern):
            try:
                path.unlink()
                logger.warning(f"Removed incompatible ChromaDB file: {path}")
            except OSError as err:
                logger.error(f"Could not remove ChromaDB file {path}: {err}")


def get_chroma_client() -> chromadb.ClientAPI:
    """Lazy-init persistent ChromaDB client."""
    global _client
    if _client is None:
        settings = get_settings()
        persist_dir = settings.CHROMA_PERSIST_DIR
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the hotel knowledge collection."""
    global _collection
    if _collection is not None:
        return _collection

    client = get_chroma_client()
    try:
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.warning(f"ChromaDB collection init failed ({e}); resetting store and retrying once.")
        _reset_chroma_store()
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_documents(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    """
    Add documents to the knowledge base.

    Args:
        documents: List of text chunks.
        metadatas: List of metadata dicts (must include hotel_id, type).
        ids: Unique IDs for each document.

    Returns:
        Number of documents added.
    """
    collection = get_collection()
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info(f"Ingested {len(documents)} documents into ChromaDB")
    return len(documents)


def query_knowledge(
    query: str,
    hotel_id: int | str,
    n_results: int = 3,
    doc_type: str | None = None,
    return_metadata: bool = False,
) -> str | dict:
    """
    Query the knowledge base for relevant information.

    Args:
        query: The user's question.
        hotel_id: Filter by hotel.
        n_results: Number of results to return.
        doc_type: Optional filter by type (policy, faq, amenity, info).

    Returns:
        If return_metadata is False: Concatenated relevant text chunks.
        If return_metadata is True: Dict with text, distances, best_score, and chunks.
    """
    if return_metadata:
        empty_res = {"text": "", "best_score": float('inf'), "distances": [], "chunks": []}
    else:
        empty_res = ""

    try:
        collection = get_collection()
    except Exception as e:
        logger.warning(f"ChromaDB unavailable, skipping RAG: {e}")
        from app.observability.metrics import metrics
        metrics.incr("chroma.query_total", outcome="unavailable")
        return empty_res

    # Build where filter — ChromaDB metadata stores hotel_id as int
    # (see sync_knowledge_base_from_mongo), so coerce the filter to the
    # same type regardless of whether callers pass str or int.
    filter_hid = int(hotel_id) if str(hotel_id).isdigit() else 111111
    where_filter: dict = {"hotel_id": filter_hid}
    if doc_type:
        where_filter["type"] = doc_type

    try:
        from app.observability.metrics import metrics
        with metrics.timer("chroma.query_latency_ms", hotel_id=hotel_id):
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )
        metrics.incr("chroma.query_total", outcome="ok", hotel_id=hotel_id)
    except Exception as e:
        # If collection is empty or query fails, return empty
        logger.warning(f"ChromaDB query failed: {e}")
        from app.observability.metrics import metrics
        metrics.incr("chroma.query_total", outcome="error", hotel_id=hotel_id)
        return empty_res

    # Concatenate results
    if not results or not results.get("documents") or not results["documents"][0]:
        return empty_res

    chunks = results["documents"][0]  # First query's results
    distances = results.get("distances", [[]])[0]
    
    best_score = distances[0] if distances else float('inf')
    concatenated_text = "\n\n---\n\n".join(chunks)

    if return_metadata:
        return {
            "text": concatenated_text,
            "best_score": best_score,
            "distances": distances,
            "chunks": chunks
        }
    
    return concatenated_text


def seed_sample_knowledge(hotel_id: str | int) -> None:
    """
    Seed the knowledge base with sample hotel information.
    Call this once during setup or testing.
    """
    sample_docs = [
        {
            "text": "Check-in time is 2:00 PM. Early check-in is subject to availability and may incur additional charges.",
            "type": "policy",
            "id": f"pol_checkin_{hotel_id}",
        },
        {
            "text": "Check-out time is 11:00 AM. Late check-out until 1:00 PM is available at 50% of the room rate. After 1:00 PM, full day charges apply.",
            "type": "policy",
            "id": f"pol_checkout_{hotel_id}",
        },
        {
            "text": "Cancellation policy: Free cancellation up to 24 hours before check-in. Cancellations within 24 hours will be charged one night's stay.",
            "type": "policy",
            "id": f"pol_cancel_{hotel_id}",
        },
        {
            "text": "Hotel amenities include: Free Wi-Fi, 24-hour front desk, room service, laundry service, parking facility, and daily housekeeping.",
            "type": "amenity",
            "id": f"amen_general_{hotel_id}",
        },
        {
            "text": "Pets are not allowed in the hotel premises. Service animals are welcome with proper documentation.",
            "type": "faq",
            "id": f"faq_pets_{hotel_id}",
        },
        {
            "text": "Children under 5 years stay free when using existing bedding. Extra bed charges apply for children above 5 years.",
            "type": "policy",
            "id": f"pol_children_{hotel_id}",
        },
        {
            "text": "Payment methods accepted: Cash, UPI, Credit Cards (Visa, Mastercard), Debit Cards, and Net Banking.",
            "type": "faq",
            "id": f"faq_payment_{hotel_id}",
        },
        {
            "text": "Airport pickup and drop service is available at additional charges. Please contact the front desk to arrange transportation.",
            "type": "faq",
            "id": f"faq_transport_{hotel_id}",
        },
    ]

    documents = [d["text"] for d in sample_docs]
    metadatas = [{"hotel_id": hotel_id, "type": d["type"]} for d in sample_docs]
    ids = [d["id"] for d in sample_docs]

    ingest_documents(documents, metadatas, ids)
    logger.info(f"Seeded {len(sample_docs)} sample knowledge docs for hotel {hotel_id}")


async def sync_knowledge_base_from_mongo(hotel_id: int | str) -> None:
    """
    Sync hotel policies, FAQs, and metadata from MongoDB into ChromaDB.
    """
    try:
        from app.db.mongodb import (
            hotel_details_collection,
            hotel_policies_collection,
            knowledge_base_collection,
        )
        kb_col = knowledge_base_collection()
        policies_col = hotel_policies_collection()
        details_col = hotel_details_collection()
    except Exception as e:
        logger.warning(f"MongoDB not connected. Skipping ChromaDB sync: {e}")
        return

    variants = [hotel_id, str(hotel_id)]
    if str(hotel_id).isdigit():
        variants.append(int(hotel_id))

    documents = []
    metadatas = []
    ids = []

    # 1. Fetch long-form Knowledge-Base articles
    kb_cursor = kb_col.find({"hotel_id": {"$in": variants}})
    async for doc in kb_cursor:
        title = doc.get("title", "")
        content = doc.get("content", "")
        if content:
            text = f"**{title}**\n{content}"
            documents.append(text)
            metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "faq", "source": "knowledge_base_articles", "title": title})
            ids.append(f"kb_article_{doc.get('_id')}")

    # 2. Fetch structured hotel policies
    pol_doc = await policies_col.find_one({"hotel_id": {"$in": variants}})
    if pol_doc:
        check_in = pol_doc.get("check_in_time") or "01:00 PM"
        check_out = pol_doc.get("check_out_time") or "11:00 AM"
        timing_text = f"**Check-in and Check-out Timings**\nCheck-in time is {check_in}. Check-out time is {check_out}. Early check-in available: {pol_doc.get('early_check_in_available', False)}. Late check-out available: {pol_doc.get('late_check_out_available', False)}."
        documents.append(timing_text)
        metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "policy", "source": "hotel_policies", "title": "Check-in and Check-out Timings"})
        ids.append(f"pol_timing_{pol_doc.get('_id')}")

        pr = pol_doc.get("property_rules") or {}
        if pr:
            accepted_ids = ", ".join(pr.get("accepted_i_ds", []))
            id_text = f"**Accepted Identification and Guest Policy**\nAccepted ID proofs: {accepted_ids or 'Aadhaar, Passport, Driving License, Govt ID'}. Same city ID allowed: {pr.get('same_city_id_allowed', False)}."
            documents.append(id_text)
            metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "policy", "source": "hotel_policies", "title": "Accepted Identification"})
            ids.append(f"pol_ids_{pol_doc.get('_id')}")

        pet = pol_doc.get("pet_policy") or {}
        if pet:
            pet_text = f"**Pet Policy**\nPets allowed: {pet.get('pets_allowed', True)}. Max pets: {pet.get('max_pets', 'One')}."
            documents.append(pet_text)
            metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "policy", "source": "hotel_policies", "title": "Pet Policy"})
            ids.append(f"pol_pet_{pol_doc.get('_id')}")

        # Property Notices & Operational Announcements
        raw_notices = pol_doc.get("notices") or pol_doc.get("Notices") or []
        for idx, n in enumerate(raw_notices):
            if isinstance(n, dict) and not n.get("expired"):
                ntitle = (
                    n.get("title")
                    or n.get("Title")
                    or n.get("category")
                    or n.get("Category")
                    or "Hotel Notice"
                ).strip()
                nnotes = (
                    n.get("notes")
                    or n.get("Notes")
                    or n.get("content")
                    or n.get("Content")
                    or ""
                ).strip()
                napplies = (
                    n.get("associated_with")
                    or n.get("associatedWith")
                    or n.get("applies_to")
                    or n.get("AppliesTo")
                    or ""
                ).strip()
                ndates = (
                    n.get("stay_dates")
                    or n.get("StayDates")
                    or n.get("start_date")
                    or n.get("startDate")
                    or ""
                ).strip()
                if n.get("end_date") and n.get("end_date") not in ndates:
                    ndates = f"{ndates} to {n.get('end_date')}".strip(" to")

                notice_text = f"**Hotel Notice and Operational Announcement: {ntitle}**\nNotice Details: {nnotes}"
                if napplies:
                    notice_text += f"\nApplies to / Location: {napplies}"
                if ndates:
                    notice_text += f"\nStay Dates / Period: {ndates}"
                documents.append(notice_text)
                metadatas.append({
                    "hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111,
                    "type": "notice",
                    "source": "hotel_policies",
                    "title": ntitle,
                })
                ids.append(f"pol_notice_{pol_doc.get('_id')}_{n.get('id', idx)}")

    # 3. Fetch hotel details & contact
    det_doc = await details_col.find_one({"hotel_id": {"$in": variants}})
    if det_doc:
        loc = det_doc.get("location") or {}
        contact = det_doc.get("contact") or {}
        if loc:
            loc_text = f"**Hotel Location and Address**\nAddress: {loc.get('property_address', '')}. Directions map link: {loc.get('google_map_link', '')}"
            documents.append(loc_text)
            metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "location", "source": "hotel_details", "title": "Location and Address"})
            ids.append(f"det_loc_{det_doc.get('_id')}")

    # 4. Fetch hotel amenities
    try:
        from app.db.mongodb import hotel_amenities_collection
        from app.repositories.hotel_repository import format_amenities
        amenities_doc = await hotel_amenities_collection().find_one({"hotel_id": {"$in": variants}})
        if amenities_doc:
            amenity_list = format_amenities(amenities_doc)
            if amenity_list:
                amenity_text = f"**Hotel Amenities and Facilities**\nAvailable amenities and facilities at Hotel: {', '.join(amenity_list)}. Free Wi-Fi is complimentary for all guests."
                documents.append(amenity_text)
                metadatas.append({"hotel_id": int(hotel_id) if str(hotel_id).isdigit() else 111111, "type": "amenities", "source": "hotel_amenities", "title": "Hotel Amenities and Facilities"})
                ids.append(f"amenities_{amenities_doc.get('_id')}")
    except Exception as e:
        logger.debug(f"Error syncing amenities to ChromaDB: {e}")

    if documents:
        int_id = int(hotel_id) if str(hotel_id).isdigit() else 111111
        try:
            collection = get_collection()
            collection.delete(where={"hotel_id": int_id})
        except Exception as e:
            logger.debug(f"Could not delete old ChromaDB entries for hotel {hotel_id}: {e}")

        ingest_documents(documents, metadatas, ids)
        logger.info(f"✅ Synced {len(documents)} knowledge docs from MongoDB for hotel {hotel_id}")
    else:
        logger.info(f"ℹ️ No knowledge base documents found in MongoDB for hotel {hotel_id}. Using default samples.")
        seed_sample_knowledge(int(hotel_id) if str(hotel_id).isdigit() else 111111)

