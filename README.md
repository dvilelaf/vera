# Vera
An autonomous [Olas](https://olas.network/) service that fact-checks controversial content.

![Vero service image](/images/logo-text.jpg)

## System requirements

- Python `>=3.8`
- [Tendermint](https://docs.tendermint.com/v0.34/introduction/install.html) `==0.34.19`
- [IPFS node](https://docs.ipfs.io/install/command-line/#official-distributions) `==0.6.0`
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Poetry](https://python-poetry.org/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

Alternatively, you can fetch this docker image with the relevant requirements satisfied:

> **_NOTE:_**  Tendermint and IPFS dependencies are missing from the image at the moment.

```bash
docker pull valory/open-autonomy-user:latest
docker container run -it valory/open-autonomy-user:latest
```


## Run you own instance

1. Clone this repo:

    ```git clone git@github.com:dvilelaf/vera.git```

2. Create the virtual environment:

    ```poetry shell && poetry install```

3. Sync packages:

    ```autonomy packages sync --update-packages```

4. Make a copy of the env file:
    ```cp sample.env .env```

5. Fill in the required environment variables.

### Run as agent

Run the script:

```bash run_agent.sh```


### Run as service

Run the script:

```bash run_service.sh```