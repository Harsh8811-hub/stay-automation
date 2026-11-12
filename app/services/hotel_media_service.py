"""Retrieve and deliver hotel images without exposing object-storage URLs to guests."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from app.db.mongodb import hotel_media_collection
from app.services.whatsapp_service import whatsapp_service

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_RESPONSE = 5
MAX_IMAGE_BYTES = 5 * 1024 * 1024
INTER_IMAGE_DELAY_SECONDS = 0.4
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png"})


@dataclass(frozen=True)
class MediaDeliveryResult:
    requested: bool
    sent_count: int = 0
    skipped_count: int = 0


def media_delivery_card(
    *,
    hotel_id: int | str | None = None,
    tags: Iterable[str] | None = None,
    aliases: Iterable[str] | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    caption: str | None = None,
    limit: int = MAX_IMAGES_PER_RESPONSE,
    is_preview: bool = False,
    explicit_request: bool = False,
) -> dict[str, Any]:
    """Build an internal-only card. It deliberately contains no storage URL."""
    normalized_tags = []
    for tag in tags or []:
        value = str(tag).strip()
        if value and value not in normalized_tags:
            normalized_tags.append(value)
            
    normalized_aliases = []
    for alias in aliases or []:
        value = str(alias).strip()
        if value and value not in normalized_aliases:
            normalized_aliases.append(value)
            
    try:
        image_limit = max(1, min(int(limit), MAX_IMAGES_PER_RESPONSE))
    except (TypeError, ValueError):
        image_limit = MAX_IMAGES_PER_RESPONSE

    return {
        "type": "media_delivery",
        "data": {
            "hotel_id": str(hotel_id) if hotel_id is not None else None,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "tags": normalized_tags,
            "aliases": normalized_aliases,
            "caption": caption or "",
            "limit": image_limit,
            "is_preview": is_preview,
            "explicit_request": explicit_request,
        },
    }


def has_media_delivery_request(cards: Iterable[Any]) -> bool:
    return any(isinstance(card, dict) and card.get("type") == "media_delivery" for card in cards)


class HotelMediaService:
    """Stateless hotel_media lookup, object download, and WhatsApp delivery service."""

    async def ensure_indexes(self) -> None:
        """Create the query index used for guest-facing media selection."""
        try:
            await hotel_media_collection().create_index(
                [
                    ("hotel_id", 1),
                    ("active", 1),
                    ("media_type", 1),
                    ("tags", 1),
                    ("entity_type", 1),
                    ("entity_id", 1),
                    ("is_primary", -1),
                    ("display_order", 1),
                ],
                name="hotel_media_delivery_lookup",
            )
        except Exception:
            logger.warning("HOTEL_MEDIA_INDEX_SETUP_FAILED", exc_info=True)

    @staticmethod
    def _score_room_type_match(
        t_item: dict[str, Any],
        entity_id: str | None,
        target_names: list[str],
        target_aliases: list[str],
    ) -> int:
        """Score candidate room type from 0 to 100 based on exactness of match."""
        t_type = str(t_item.get("type") or "").strip()
        t_room_id = str(t_item.get("roomId") or t_item.get("entity_id") or "").strip()

        if not t_type or t_type.lower() == "hotel":
            return 0

        # 1. Exact Room ID match
        if entity_id and t_room_id:
            if entity_id.strip().lower() == t_room_id.strip().lower():
                return 100

        generic_words = frozenset({
            "room", "rooms", "bed", "beds", "ac", "non-ac", "non", "hotel"
        })
        distinctive_stops = frozenset({
            "deluxe", "standard", "superior", "classic", "executive", "luxury", "comfort"
        })

        norm_t_type = re.sub(r"[^\w\s]", " ", t_type.lower()).strip()
        norm_t_type = " ".join(norm_t_type.split())
        t_words = set(norm_t_type.split())

        best_score = 0
        for name in target_names:
            norm_name = re.sub(r"[^\w\s]", " ", (name or "").lower()).strip()
            norm_name = " ".join(norm_name.split())
            if not norm_name:
                continue

            name_words = set(norm_name.split())

            # Exact normalized string match
            if norm_name == norm_t_type:
                return 100

            # Exact word set match (e.g. 'King Bed Deluxe Room' vs 'Deluxe Room King Bed')
            if name_words == t_words:
                best_score = max(best_score, 95)
                continue

            # Token overlap excluding generic stop words
            n_spec = name_words - generic_words
            t_spec = t_words - generic_words

            if n_spec and t_spec:
                if n_spec == t_spec:
                    best_score = max(best_score, 90)
                    continue

                overlap = n_spec & t_spec
                distinctive_overlap = overlap - distinctive_stops
                if distinctive_overlap:
                    if n_spec == overlap or t_spec == overlap:
                        best_score = max(best_score, 80)

        # 3. Check aliases ONLY if no strong target_name match
        if best_score < 80:
            for alias in target_aliases:
                norm_alias = re.sub(r"[^\w\s]", " ", (alias or "").lower()).strip()
                norm_alias = " ".join(norm_alias.split())
                if not norm_alias:
                    continue
                # Aliases must NOT be a single generic word like 'deluxe' or 'room'
                if norm_alias in distinctive_stops or norm_alias in generic_words:
                    continue
                alias_words = set(norm_alias.split())
                if alias_words == t_words:
                    best_score = max(best_score, 85)
                elif norm_alias == norm_t_type:
                    best_score = max(best_score, 85)

        return best_score

    NON_BEDROOM_TAGS: frozenset[str] = frozenset({
        "washroom", "bathroom", "toilet", "bath", "restroom", "wc",
        "amenity", "amenities", "reception", "lobby", "facade", "entrance",
        "elevator", "lift", "pool", "swimming pool", "gym", "fitness",
        "restaurant", "dining", "bar", "hall", "banquet", "corridor",
        "outside view", "exterior", "parking", "garden", "activities",
        "stairs", "staircase", "terrace", "balcony",
    })

    @classmethod
    def _extract_photos_from_item(cls, t_item: dict[str, Any], is_room: bool = True) -> list[dict[str, Any]]:
        photos = t_item.get("photos", [])
        if not isinstance(photos, list):
            return []

        bedroom_photos: list[dict[str, Any]] = []
        other_photos: list[dict[str, Any]] = []

        for p in photos:
            if not isinstance(p, dict):
                continue
            gcs_url = p.get("gcs_url")
            if not isinstance(gcs_url, str) or not gcs_url.strip():
                continue

            photo_obj = {
                "gcs_url": gcs_url.strip(),
                "mime_type": "image/jpeg",
                "content_type": "image/jpeg",
                "caption": p.get("tag") or t_item.get("type") or "",
                "is_primary": bool(p.get("is_cover", False)),
            }

            p_tag = str(p.get("tag") or "").lower().strip()
            if is_room:
                is_non_bedroom = any(feat in p_tag for feat in cls.NON_BEDROOM_TAGS)
                if is_non_bedroom:
                    other_photos.append(photo_obj)
                else:
                    bedroom_photos.append(photo_obj)
            else:
                bedroom_photos.append(photo_obj)

        # STRICT BEDROOM FILTER: For room queries, return ONLY bedroom photos.
        # NEVER fall back to washrooms, bathrooms, toilets, or general amenities.
        if is_room:
            results = bedroom_photos
        else:
            results = bedroom_photos or other_photos

        results.sort(key=lambda x: not x.get("is_primary", False))
        return results

    async def _find_nested_images(
        self,
        *,
        hotel_id: int | str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        tags: Iterable[str] | None = None,
        aliases: Iterable[str] | None = None,
        limit: int = MAX_IMAGES_PER_RESPONSE,
    ) -> list[dict[str, Any]]:
        str_hid = str(hotel_id)
        if str_hid in ("988273", "108959"):
            hotel_variants: list[Any] = ["108959", "988273", 108959, 988273]
        else:
            hotel_variants = [hotel_id, str_hid]
            if str_hid.isdigit():
                hotel_variants.append(int(str_hid))
            hotel_variants = list(dict.fromkeys(hotel_variants))

        try:
            doc = await hotel_media_collection().find_one({
                "hotel_id": {"$in": hotel_variants},
                "types": {"$exists": True}
            })
        except Exception:
            return []

        if not doc or "types" not in doc:
            return []

        # Gather target candidate names for room matching
        target_names: list[str] = []
        for t in (tags or []):
            if str(t).strip():
                target_names.append(str(t).strip())

        # If entity_id is provided, try resolving canonical room name from room_types collection
        if entity_id:
            try:
                from app.db.mongodb import room_types_collection
                rt_doc = await room_types_collection().find_one({
                    "hotel_id": {"$in": hotel_variants},
                    "$or": [
                        {"room_type_id": entity_id},
                        {"roomId": entity_id},
                        {"room_id": entity_id},
                        {"_id": entity_id},
                    ]
                })
                if rt_doc and (rt_doc.get("name") or rt_doc.get("room_name")):
                    c_name = rt_doc.get("name") or rt_doc.get("room_name")
                    if c_name not in target_names:
                        target_names.insert(0, c_name)
            except Exception:
                pass

        target_aliases: list[str] = []
        for a in (aliases or []):
            if str(a).strip():
                target_aliases.append(str(a).strip())

        types_list = [t for t in doc.get("types", []) if isinstance(t, dict)]

        # --- CASE 1: ROOM PHOTO LOOKUP ---
        is_room_lookup = (entity_type == "room") or bool(target_names) or bool(entity_id)

        if is_room_lookup:
            best_score = 0
            best_item = None

            for t_item in types_list:
                score = self._score_room_type_match(
                    t_item=t_item,
                    entity_id=entity_id,
                    target_names=target_names,
                    target_aliases=target_aliases,
                )
                if score > best_score:
                    best_score = score
                    best_item = t_item

            if best_item and best_score >= 70:
                room_photos = self._extract_photos_from_item(best_item, is_room=True)
                if room_photos:
                    return room_photos[:limit]

            # If room photos were requested but not matched with high confidence,
            # NEVER return another different room's photos
            if entity_type == "room":
                return []

        # --- CASE 2: AMENITY / GENERAL / FACADE / LOBBY LOOKUP ---
        feature_tags = {
            "washroom", "bathroom", "reception", "lobby", "facade",
            "entrance", "pool", "gym", "restaurant", "dining", "balcony"
        }
        requested_tags = {t.lower() for t in target_names + target_aliases}

        tag_matched_photos: list[dict[str, Any]] = []
        hotel_general_photos: list[dict[str, Any]] = []

        for t_item in types_list:
            t_type = str(t_item.get("type") or "").strip().lower()
            photos = t_item.get("photos", [])
            if not isinstance(photos, list):
                continue

            for p in photos:
                if not isinstance(p, dict):
                    continue
                gcs_url = p.get("gcs_url")
                if not isinstance(gcs_url, str) or not gcs_url.strip():
                    continue

                photo_obj = {
                    "gcs_url": gcs_url.strip(),
                    "mime_type": "image/jpeg",
                    "content_type": "image/jpeg",
                    "caption": p.get("tag") or t_item.get("type") or "",
                    "is_primary": bool(p.get("is_cover", False)),
                }

                p_tag = str(p.get("tag") or "").lower()
                if any(ft in p_tag for ft in requested_tags if ft in feature_tags):
                    tag_matched_photos.append(photo_obj)
                elif t_type == "hotel":
                    hotel_general_photos.append(photo_obj)

        results = tag_matched_photos or hotel_general_photos
        return results[:limit]

    async def find_images(
        self,
        *,
        hotel_id: int,
        entity_type: str | None = None,
        entity_id: str | None = None,
        tags: Iterable[str] | None = None,
        aliases: Iterable[str] | None = None,
        limit: int = MAX_IMAGES_PER_RESPONSE,
    ) -> list[dict[str, Any]]:
        """Return active image records in guest-facing priority order."""
        # 1. Try real hotel photos from nested document structure first
        nested_images = await self._find_nested_images(
            hotel_id=hotel_id,
            entity_type=entity_type,
            entity_id=entity_id,
            tags=tags,
            aliases=aliases,
            limit=limit,
        )
        if nested_images:
            return nested_images

        query: dict[str, Any] = {
            "hotel_id": hotel_id,
            "active": True,
            "media_type": "image",
        }
        if entity_type:
            query["entity_type"] = entity_type

        # Strictly exclude non-bedroom photos when querying rooms
        if entity_type == "room":
            neg_regex = re.compile(r"washroom|bathroom|toilet|bath|restroom|amenit|reception|lobby|facade|entrance|pool|gym", re.I)
            query["tags"] = {"$nin": [neg_regex]}
            
        or_conditions = []
        if entity_id:
            or_conditions.append({"entity_id": entity_id})

        # Combine tags and aliases for matching against DB fields
        all_terms = []
        for t in (tags or []):
            if str(t).strip(): all_terms.append(str(t).strip())
        for a in (aliases or []):
            if str(a).strip(): all_terms.append(str(a).strip())
            
        if all_terms:
            tag_patterns = [re.compile(rf"^{re.escape(term)}$", re.IGNORECASE) for term in all_terms]
            or_conditions.append({"tags": {"$in": tag_patterns}})
            or_conditions.append({"aliases": {"$in": tag_patterns}})
            or_conditions.append({"entity_id": {"$in": tag_patterns}})

        if or_conditions:
            query["$or"] = or_conditions

        metadata_limit = max(limit * 3, limit)
        
        # Use aggregation to sort missing display_order to the end
        pipeline = [
            {"$match": query},
            {"$addFields": {
                "display_order_sort": {"$ifNull": ["$display_order", 99999]}
            }},
            {"$sort": {"is_primary": -1, "display_order_sort": 1}},
            {"$limit": metadata_limit},
            {"$project": {"gcs_url": 1, "mime_type": 1, "content_type": 1, "is_primary": 1, "display_order": 1}}
        ]
        
        cursor = hotel_media_collection().aggregate(pipeline)
        return await cursor.to_list(length=metadata_limit)

    async def deliver_card_requests(
        self,
        *,
        cards: Iterable[Any],
        to_number: str,
        hotel_id: int,
        sender_id: str | None = None,
    ) -> MediaDeliveryResult:
        """Resolve and send every media_delivery card without affecting text delivery."""
        requested = False
        sent_count = 0
        skipped_count = 0

        for card in cards:
            if not isinstance(card, dict) or card.get("type") != "media_delivery":
                continue
            remaining = MAX_IMAGES_PER_RESPONSE - sent_count
            if remaining <= 0:
                break
            requested = True
            data = card.get("data") or {}
            try:
                card_limit = max(1, min(int(data.get("limit", MAX_IMAGES_PER_RESPONSE)), remaining))
            except (TypeError, ValueError):
                card_limit = remaining
            if sent_count:
                await asyncio.sleep(INTER_IMAGE_DELAY_SECONDS)
            result = await self.deliver_images(
                hotel_id=hotel_id,
                to_number=to_number,
                sender_id=sender_id,
                entity_type=data.get("entity_type"),
                entity_id=data.get("entity_id"),
                tags=data.get("tags") or [],
                caption=data.get("caption") or "",
                limit=card_limit,
            )
            sent_count += result.sent_count
            skipped_count += result.skipped_count

        return MediaDeliveryResult(requested=requested, sent_count=sent_count, skipped_count=skipped_count)

    async def deliver_images(
        self,
        *,
        hotel_id: int,
        to_number: str,
        sender_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        tags: Iterable[str] | None = None,
        caption: str = "",
        limit: int = MAX_IMAGES_PER_RESPONSE,
    ) -> MediaDeliveryResult:
        """Download selected images in memory and send them through the WhatsApp media API."""
        try:
            image_limit = max(1, min(int(limit), MAX_IMAGES_PER_RESPONSE))
        except (TypeError, ValueError):
            image_limit = MAX_IMAGES_PER_RESPONSE
        try:
            records = await self.find_images(
                hotel_id=hotel_id,
                entity_type=entity_type,
                entity_id=entity_id,
                tags=tags,
                limit=image_limit,
            )
        except Exception:
            logger.exception("HOTEL_MEDIA_LOOKUP_FAILED | hotel_id=%s", hotel_id)
            return MediaDeliveryResult(requested=True, skipped_count=1)

        sent_count = 0
        skipped_count = 0
        seen_urls: set[str] = set()
        timeout = httpx.Timeout(8.0, connect=5.0)

        logger.info(
            "HOTEL_MEDIA_DELIVER_START | hotel_id=%s records=%s limit=%s to=…",
            hotel_id,
            len(records),
            image_limit,
        )

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for record in records:
                if sent_count >= image_limit:
                    break

                gcs_url = record.get("gcs_url")
                if not isinstance(gcs_url, str) or not gcs_url.strip():
                    logger.warning("HOTEL_MEDIA_SKIPPED | hotel_id=%s reason=missing_gcs_url", hotel_id)
                    skipped_count += 1
                    continue
                if gcs_url in seen_urls:
                    skipped_count += 1
                    continue
                seen_urls.add(gcs_url)

                try:
                    downloaded = await self._download_image(client, gcs_url, record)
                except Exception:
                    logger.exception("HOTEL_MEDIA_DOWNLOAD_FAILED")
                    skipped_count += 1
                    continue
                if not downloaded:
                    skipped_count += 1
                    continue
                raw_cap = caption or record.get("caption") or ""
                image_caption = self.sanitize_category_caption(raw_cap)
                if sent_count:
                    await asyncio.sleep(INTER_IMAGE_DELAY_SECONDS)
                try:
                    sent = await asyncio.to_thread(
                        whatsapp_service.send_image_bytes,
                        to_number=to_number,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        filename=filename,
                        caption=image_caption,
                        sender_id=sender_id,
                        hotel_id=hotel_id,
                    )
                except Exception:
                    logger.exception("WHATSAPP_IMAGE_DELIVERY_FAILED | hotel_id=%s", hotel_id)
                    skipped_count += 1
                    continue
                if sent:
                    sent_count += 1
                else:
                    skipped_count += 1

        logger.info(
            "HOTEL_MEDIA_DELIVER_DONE | hotel_id=%s sent=%s skipped=%s",
            hotel_id,
            sent_count,
            skipped_count,
        )
        return MediaDeliveryResult(requested=True, sent_count=sent_count, skipped_count=skipped_count)

    @staticmethod
    def sanitize_category_caption(raw_caption: str) -> str:
        """
        T1 — Photo message: Category name only.
        No rate, no Rs, no /-, no estimate, no totals, no availability text.
        One caption = one category name. Example: 'Deluxe Room'.
        """
        if not raw_caption:
            return ""
        first_line = str(raw_caption).split("\n")[0].strip()
        # Strip trailing parenthesized price details like (Rs 2,500/night) or (₹3,000)
        cleaned = re.sub(r"\s*\([^)]*(?:₹|rs\.?|inr|/-|\d+)[^)]*\)", "", first_line, flags=re.IGNORECASE).strip()
        # Strip price suffix patterns like ' - Rs 2,500', ' @ ₹3,000', ' : 2500/-', ' / night'
        cleaned = re.sub(
            r"(?:[-–—|:@]\s*)?(?:₹|rs\.?|inr|\/)\s*[\d,]+.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # Also strip trailing numeric rates with '/-' or 'per night'
        cleaned = re.sub(
            r"(?:[-–—|:@]\s*)?\b\d{3,6}\b\s*(?:\/-|\s*per\s*night|\s*\/night).*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # Strip leading emojis, symbols, markdown asterisks/underscores
        cleaned = re.sub(r"^[📸🏨🛏️✨*_\s]+", "", cleaned)
        # Strip trailing punctuation and markdown formatting
        cleaned = re.sub(r"[*_\s\-–—:]+$", "", cleaned)
        if cleaned.lower().startswith("photos of "):
            cleaned = cleaned[10:].strip()
        return cleaned.strip("*_`'\" ")

    async def _download_image(
        self,
        client: httpx.AsyncClient,
        gcs_url: str,
        record: dict[str, Any],
    ) -> tuple[bytes, str, str] | None:
        """Bound a download in memory and reject non-image or oversized content."""
        try:
            async with client.stream("GET", gcs_url) as response:
                response.raise_for_status()
                mime_type = (response.headers.get("content-type") or record.get("mime_type") or record.get("content_type") or "").split(";", 1)[0].lower()
                if mime_type not in SUPPORTED_IMAGE_TYPES:
                    logger.warning("HOTEL_MEDIA_SKIPPED | reason=unsupported_mime mime_type=%s", mime_type or "missing")
                    return None

                image = bytearray()
                async for chunk in response.aiter_bytes():
                    image.extend(chunk)
                    if len(image) > MAX_IMAGE_BYTES:
                        logger.warning("HOTEL_MEDIA_SKIPPED | reason=image_too_large")
                        return None
        except Exception:
            logger.warning("HOTEL_MEDIA_DOWNLOAD_FAILED", exc_info=True)
            return None

        filename = PurePosixPath(urlparse(gcs_url).path).name or "hotel-image"
        return bytes(image), mime_type, filename

    async def resolve_media_cards_for_api(self, hotel_id: int, cards: list[Any]) -> list[Any]:
        """Resolve media_delivery cards into image cards with URLs for the CRM REST API."""
        resolved_cards = []
        for card in cards or []:
            if not isinstance(card, dict) or card.get("type") != "media_delivery":
                resolved_cards.append(card)
                continue
                
            data = card.get("data") or {}
            try:
                records = await self.find_images(
                    hotel_id=hotel_id,
                    entity_type=data.get("entity_type"),
                    entity_id=data.get("entity_id"),
                    tags=data.get("tags"),
                    limit=data.get("limit", MAX_IMAGES_PER_RESPONSE),
                )
            except Exception:
                logger.exception("Failed to resolve media for API | hotel_id=%s", hotel_id)
                records = []
                
            caption = self.sanitize_category_caption(data.get("caption") or "")
            for i, record in enumerate(records):
                gcs_url = record.get("gcs_url")
                if gcs_url:
                    if "images.unsplash.com" in gcs_url:
                        gcs_url = gcs_url.replace("auto=format", "fm=jpg")
                    logger.info("ROOM_IMAGE_SENT | url=%s", gcs_url)
                    rec_cap = self.sanitize_category_caption(record.get("caption") or "")
                    resolved_cards.append({
                        "type": "image",
                        "url": gcs_url,
                        "caption": caption or rec_cap
                    })
        return resolved_cards


hotel_media_service = HotelMediaService()
