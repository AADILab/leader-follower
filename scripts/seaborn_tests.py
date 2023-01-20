import seaborn as sns
import seaborn.objects as so
from matplotlib.pyplot import show

sns.set_theme()

tips = sns.load_dataset("tips")

print(tips)

p = so.Plot(tips, "total_bill", "tip").add(so.Dot())
p.show()

# sns.relplot(
#     data=tips,
#     x="total_bill", y="tip", col="time",
#     hue="smoker", style="smoker", size="size",
#     kind="line"
# )

# show()
