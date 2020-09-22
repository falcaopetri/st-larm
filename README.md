<h1 align="center">Welcome to ST-LARM 👋</h1>
<p>
</p>

> Semantic Trajectories - Logical Association Rules Mining

This repository is the companion code and experiment to the paper _Towards logical association rule mining on ontology-based semantic trajectories_ accepted to _ICMLA 2020_.

## Pipeline

![Data pipeline](./pipeline.png)

1. Semantic Trajectories: the result of trajectory data semantic enrichment;
2. Ontology-based representation: the result of representing data in an ontology-based representation, such as [STEP](http://purl.org/net/step);
3. Application-specific representation: previous representation converted in an application-specific one. See [stlarm/README.md](./stlarm/README.md) for pseudocode of this process;
4. Mined Rules: the set of rules mined by AMIE 3;
5. Metarules: the set of metarules built from mined rules.

See [experiments/nyc-foursquare/README.md](./experiments/nyc-foursquare/README.md) for a concrete pipeline example.

## Repository structure

- `stlarm`: Contains all code related to ST-LARM. This includes data manipulation, ontology processing, and result analysis utilities.
- `experiments`: Pipeline execution, data generated, and analyses.
- `amie`: AMIE 3 code + our modifications. For simplicity, we make available in this repo only the executable file. If you need to check the code, use the `icmla2020` branch at [falcaopetri#amie](https://github.com/falcaopetri/amie/tree/icmla2020).
- `trajminer`: Auxiliary library for trajectory data manipulation.
- `ontology-visualization`: Auxiliary library for graphviz-based ontology visualization.

## Setup

This step is provided if you want to execute or reproduce some experiments. 
If you want only to see or have some basic interaction with data or results, you can use GitHub's Jupyter Notebook rendering and CSV browsing by just navigating to the `experiments/` folder.

A docker container provides everything you need. Be sure you have [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/) installed, and then execute the following command:

```sh
sudo docker-compose up
```

By accessing [`http://localhost:8888`](http://localhost:8888) in your browser, you will have access to a Jupyter Lab environment.


## Authors
- Antonio Carlos Falcão Petri;
- Diego Furtado Silva.