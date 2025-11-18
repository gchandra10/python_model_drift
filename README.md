```
uv sync

source .venv/bin/activate

source .venv\Scritpts\activate.bat

```
**Start MlFlow Server**

```
mlflow server --host 127.0.0.1 --port 8080 \
--backend-store-uri sqlite:///mlflow.db 
```


**How to test?**

**Run initial model builder**

```
uv run python train_register.py
```

**Run Drift Monitor**

```
uv run python monitor_drift.py
```

