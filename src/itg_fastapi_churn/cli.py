import uvicorn

APP_PATH = "itg_fastapi_churn.main:app"


def main() -> None:
    """
    Run the uvicorn web server with the service application
    """
    uvicorn.run(APP_PATH, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
