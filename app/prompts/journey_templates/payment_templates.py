def payment_link(payment_url: str, total: float, vpa: str = None, qr_code_url: str = None, **kwargs) -> list[dict]:
    from app.utils.template_renderers import button_card
    text = (
        f"Great! Here's your secure payment link:\n\n"
        f"💳 *Total: ₹{total:,.0f}*\n"
        f"🔗 {payment_url}\n\n"
        f"Once payment is completed, your booking will be confirmed immediately! 🎉\n"
        f"_This link is valid for 15 minutes._"
    )
    cards = button_card(text, [{"id": "payment_paid", "title": "I've paid"}])
    if qr_code_url:
        caption = f"Scan to pay advance ₹{total:,.0f} via any UPI App (VPA: {vpa})" if vpa else "Scan to pay advance via any UPI App"
        cards.append({
            "type": "image",
            "url": qr_code_url,
            "image_url": qr_code_url,
            "media_url": qr_code_url,
            "caption": caption,
        })
    return cards

def payment_failed(link: str = None, prepend_text: str = None, **kwargs) -> list[dict]:
    link_str = f"using this link:\n🔗 {link}" if link else "using the link shared earlier."
    msg = "Your payment is still pending. ⏳\n\n"
    if prepend_text:
        msg = f"{prepend_text}\n\n{msg}"
    return [{"type": "text", "text": f"{msg}Please complete the payment {link_str}\n\n"
                                     f"Payment complete hote hi booking turant confirm ho jayegi."}]
