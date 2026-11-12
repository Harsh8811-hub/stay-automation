"""Provider-neutral LLM access.

`compat` is a `google.generativeai`-shaped facade that dispatches to whichever
provider `LLM_PROVIDER` names. Call sites import it as `genai` and do not change:

    from app.services.llm import compat as genai

    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    response = await model.generate_content_async([prompt, image_part])

WHY A FACADE INSTEAD OF REWRITING EVERY CALL SITE
The bot reaches the model from fifteen places across thirteen modules, each
with its own prompt, JSON parsing and error handling that have been debugged
against real guests. A facade keeps every one of those intact and makes the
provider a deployment decision rather than a refactor. When a second provider
needs a capability the facade cannot express, change that call site, not all of
them.
"""

from app.services.llm import genai_compat as compat

genai = compat
__all__ = ["compat", "genai"]
