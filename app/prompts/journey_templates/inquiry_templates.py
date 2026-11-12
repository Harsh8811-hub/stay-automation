def inq_greet(hotel_name: str = "our hotel", **kwargs) -> list[dict]:
    return [{"type": "text", "text": f"*Namaste! 🙏 Welcome to {hotel_name}.*\n\n"
                                     "I'm your Hotel assistant. How can I help you today?\n\n"
                                     "Please choose an option or just type your question:\n\n"
                                     "1 *Make a new booking*\n"
                                     "2 *Check my existing booking*\n"
                                     "3 *Room prices & availability*\n"
                                     "4 *Hotel location & directions*\n"
                                     "5 *Room photos & amenities*\n"
                                     "6 *Something else*"}]
