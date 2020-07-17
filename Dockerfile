FROM jupyter/scipy-notebook:5197709e9f23

USER root

RUN apt-get update

# Install java: https://stackoverflow.com/a/44058196
# Install OpenJDK-8
RUN apt-get install -y openjdk-8-jdk ca-certificates-java && \
    # Fix certificate issues
    update-ca-certificates -f

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN apt-get install -y graphviz

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*;

ADD trajminer/requirements.txt requirements_trajminer.txt
RUN pip install -r requirements_trajminer.txt

ADD ontology-visualization/requirements.txt requirements_ontviz.txt
RUN pip install -r requirements_ontviz.txt

ADD stlarm/requirements.txt requirements_stlarm.txt
RUN pip install -r requirements_stlarm.txt
