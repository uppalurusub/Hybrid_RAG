from app.core.config import settings
def tracing_config(operation, tags=None):
    return {"run_name":operation,"tags":tags or ["component:rag"],
            "metadata":{"operation":operation,"application":settings.app_name}}
