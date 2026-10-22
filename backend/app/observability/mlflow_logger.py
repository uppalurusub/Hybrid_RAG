import logging
import mlflow
from app.core.config import settings
log = logging.getLogger(__name__)
def log_metrics(metrics, params=None, tags=None):
    if not settings.mlflow_enabled: return None
    try:
        
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run() as run:
            for k,v in (params or {}).items():
                if v is not None: mlflow.log_param(k, str(v)[:500])
            for k,v in (tags or {}).items():
                if v is not None: mlflow.set_tag(k, str(v)[:500])
            for k,v in metrics.items():
                if isinstance(v,(int,float)) and v is not None: mlflow.log_metric(k,float(v))
            return run.info.run_id
    except Exception:
        log.exception("MLflow metrics logging failed")
        return None
