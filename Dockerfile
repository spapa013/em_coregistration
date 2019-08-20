# FROM ninai/stimulus-pipeline # Will only work if you have access, want to be independent from this and all of its tables though. Meaning copy what we need to our schemas.
FROM ninai/pipeline:base
LABEL maintainer="Christos & Stelios Papadopoulos, et al."

RUN apt-get -y update && \
    apt-get -y install graphviz libxml2-dev python3-cairosvg parallel fish
    
RUN pip3 install datajoint --upgrade

WORKDIR /src

RUN pip3 install scikit-build

RUN pip3 install python-igraph xlrd
# pyvista version 0.20.1 is required because newer pyvista versions break the build
RUN pip3 install pyvista==0.20.1
# RUN pip3 install meshparty
RUN pip3 install pykdtree tables
# RUN pip3 install analysisdatalink
# RUN pip3 install -e git+https://github.com/seung-lab/AnalysisDataLink.git#egg=analysisdatalink
# RUN pip3 install -e git+https://github.com/seung-lab/cloud-volume.git@graphene#egg=cloud-volume
RUN apt-get -y install libassimp-dev

RUN pip3 install ipyvolume jupyterlab statsmodels pycircstat nose autograd torch
RUN pip3 install seaborn --upgrade
RUN pip3 install jgraph
RUN pip3 install marshmallow==2.19.5

ADD . /src/em2p_coreg
RUN pip3 install -e /src/em2p_coreg/python

WORKDIR /notebooks

RUN mkdir -p /scripts
ADD ./jupyter/run_jupyter.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter.sh"]