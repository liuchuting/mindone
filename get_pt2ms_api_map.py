import pandas as pd

# 读取表格
df = pd.read_excel("interface_support.xlsx")  # 或 pd.read_csv()

# 用列索引替代列名
# 0:接口列，1:结论列，2:接口修正列
mint_nn_map = {}
ops_map = {}
p2m_map = {}


url = "https://example.com/page-with-table.html"
dfs = pd.read_html(url)  # 读取网页中所有表格，返回一个列表

df = dfs[0]  # 取第一个表格，或者你根据情况选其他表格
print(df.head())


# 过滤结论列（第1索引）为“支持”的行
df_supported = df[df.iloc[:, 1] == "支持"]

for _, row in df_supported.iterrows():
    torch_iface = str(row.iloc[0]).strip()     # 接口列
    corrections = str(row.iloc[2]).strip()     # 接口修正列

    if not corrections:
        continue

    # 解析接口修正列，取第一个值（逗号/分号/空格分割）
    ms_iface = corrections.split('\n')[0].strip()

    # 分类存map
    if "mint.nn" in ms_iface and "torch.nn" in torch_iface:
        mint_nn_map[torch_iface.strip("torch")] = ms_iface.strip("mindspore")
    elif "ops" in ms_iface:
        ops_map[torch_iface] = ms_iface
    else:
        p2m_map[torch_iface] = ms_iface

print("mint_nn_map:", mint_nn_map)
print("ops_map:", ops_map)
print("p2m_map:", p2m_map)
