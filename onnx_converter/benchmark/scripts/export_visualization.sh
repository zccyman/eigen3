#!/usr/bin/env bash

work_dir=$1

mmdc -i ${work_dir}/test_vis.mmd -o ${work_dir}/test_vis.mmd.pdf --width 1024000 --height 1024000 --scale 1 --puppeteerConfigFile ${work_dir}/puppeteer-config.json --configFile ${work_dir}/mermaidRenderConfig.json

python benchmark/scripts/delete_blank_pages.py --export_dir ${work_dir}

# docker run -it \
# -v ${work_dir}:/data/work_dir \
# minlag/mermaid-cli \
# -i /data/work_dir/test_vis.mmd \
# -o /data/work_dir/test_vis.pdf \
# --configFile="/data/work_dir/mermaidRenderConfig.json"

# for container in `docker container ls -a | grep minlag/mermaid-cli  | awk '{print $1}'`;
# do 
# docker rm ${container}
# done