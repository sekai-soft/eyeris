# eyeris
Simple media recommendation engine

## Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Running the App

Start the development server:

```bash
uv run fastapi dev app.py
```

The app will be available at `http://localhost:8000`

For production:

```bash
uv run fastapi run app.py
```

### API Documentation

Once the server is running, you can access:
- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`
