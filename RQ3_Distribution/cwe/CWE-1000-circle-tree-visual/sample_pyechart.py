from pyecharts import options as opts
from pyecharts.charts import Tree
import json

def dict_del(key,obj):
    if isinstance(obj, dict):
        if key in obj:
            obj.pop(key)
        for k, v in obj.items():
            dict_del(key,v)
    if isinstance(obj, list):
        for x in obj:
            dict_del(key,x)
    else:
        pass


file='./sample_cwe.json'
with open(file) as f:
 data=json.load(f)
 dict_del("parents",data)
 print(data)


c = (
    Tree()
    .add("",
        data=[data],
        # pos_top="10%",
        # pos_bottom="20%",
        layout="radial",
        #symbol="emptyCircle",
        #orient='center',
        initial_tree_depth=2,  # initial_tree_depth set to 2 to expand first two layers
        symbol_size=30,)
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove")
        # label_opts=opts.LabelOpts(font_size=14)
    )
    .render("./cwe_tree_test.html")
)
