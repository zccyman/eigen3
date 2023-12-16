#!/bin/bash

root_dir=$1
export_dir=$2
passwd=$3

sudo -S mv ${root_dir}/work_dir/weights ${export_dir} << EOF 
${passwd}
EOF

sudo -S mv ${root_dir}/work_dir/process.log ${export_dir} << EOF 
${passwd}
EOF

sudo -S mv ${root_dir}/work_dir/export.log ${export_dir} << EOF 
${passwd}
EOF

sudo -S mv ${root_dir}/work_dir/model.c ${export_dir} << EOF 
${passwd}
EOF

sudo -S mv ${root_dir}/work_dir/test_vis.mmd ${export_dir} << EOF 
${passwd}
EOF