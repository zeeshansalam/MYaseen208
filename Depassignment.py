

## 2017 AG 3669 Zeeshan Salam

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as numpy
import pandas as pandas
from matplotlib import cm

##ResultImg of NoisedImg for Y,X1,X2
NoisedImg = [5, 7, 15, 17, 29, 16]
FilterImg = [0, 0, 10, 10, 25, 25]
ResultImg = [0, 0, 100, 100, 200, 300]

##Find degree of fredom for all variables
degf = pandas.DataFrame(
    {
        "NoisedImg": NoisedImg
        , "FilterImg": FilterImg
        , "ResultImg": ResultImg
    }
)

print(degf)
## Apply regression formula and draw result  in 3d graphs
Reg4 = ols(formula="ResultImg ~ NoisedImg + FilterImg", NoisedImg=degf)
Fit4 = Reg4.fit()
print(Fit4.summary())
print(Fit4.params)
print(Fit4.fittedvalues)
print(Fit4.resid)
print(Fit4.bse)
print(Fit4.centered_tss)
print(anova_lm(Fit4))
fg = plt.figure()
ax = fg.add_subplot(111, projection="3d")
ax.scatter(
    degf["NoisedImg"]
    , degf["FilterImg"]
    , degf["ResultImg"]
    , color="Blue"
    , marker="+"
    , alpha=0.5
)
##Draw the axis for all values with alpha values changes
ax.set_xlabel("NoisedImg")
ax.set_ylabel("FilterImg")
ax.set_zlabel("ResultImg")
x_surf = numpy.arange(110, 700, 40)
y_surf = numpy.arange(20, 30, 2)
x_surf, y_surf = numpy.meshgrid(x_surf, y_surf)

exog = pandas.core.frame.NoisedImgFrame({
    "NoisedImg": x_surf.ravel()
    , "FilterImg": y_surf.ravel()
})
out = Fit4.predict(exog=exog)
ax.plot_surface(
    x_surf
    , y_surf
    , out.values.reshape(x_surf.shape)
    , rstride=5
    , cstride=5
    , color="Orange"
    , alpha=0.5
)
plt.show()

fg = plt.figure()
ax = fg.add_subplot(111, projection="3d")
ax.scatter(
    degf["NoisedImg"]
    , degf["FilterImg"]
    , degf["ResultImg"]
    , color="Red"
    , marker="+"
    , alpha=0.5
)
## Draw all the vlaues for all axis in plot
ax.set_xlabel("NoisedImg")
ax.set_ylabel("FilterImg")
ax.set_zlabel("ResultImg")

plt.show()

