import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

import random



from scipy.stats import multivariate_normal
from scipy import stats
import seaborn as sns



num_conn = 200


a_mean = 1.2e-7
b_mean = 5.6e-7
a_var = (4.8e-8)**0.5
b_var = (2.4e-7)**0.5


a_var = 0.5
b_var = 0.5



material_a_min = 33
material_a_max = 170
material_b_min = 280
material_b_max = 960


corr = 1
# """

"""x = np.random.uniform(size=num_conn)
y = np.random.uniform(size=num_conn)

if corr == 0:

    a = x
    b = y

elif corr == 1:

    a = x
    b = x

elif corr == -1:

    a = x
    b = 1 - x

else:"""

"""
Generating normal distributions with correlations, and converting them into
a uniform distribution!

I.e. using a gausian copula created using multivariate normal
https://twiecki.io/blog/2018/05/03/copulas/

Copulas allow us to decompose a joint probability distribution into their
marginals (which by definition have no correlation) and a function which
couples (hence the name) them together and thus allows us to specify the
correlation seperately. The copula is that coupling function.

Often this is used for:
    - Generate correlated normal distributions
    - Decompose/convert to uniform distribution
    - Convert uniform to any desired distribution

"""

# # same
x = np.random.multivariate_normal([0, 0], [[1., corr], [corr, 1.]], size=num_conn)
print("good")

mvnorm = multivariate_normal([0, 0], [[1, corr], [corr, 1]], allow_singular=True)
x = mvnorm.rvs((num_conn,))
print("good 2")

#print("\n\n", sns.__version__)


# # plot correlated normal distributions
h = sns.jointplot(x=x[:, 0], y=x[:, 1]+0.00001*np.random.rand(len(x[:,0])), kind='kde')
h.set_axis_labels('X1', 'X2', fontsize=16)
h.ax_joint.plot(x[:, 0], x[:, 1], ".", color="#800000")
h.ax_marg_y.set_ylim(-np.max(np.abs(x[:, 1]))-0.5, np.max(np.abs(x[:, 1]))+0.5)
h.ax_marg_x.set_xlim(-np.max(np.abs(x[:, 0]))-0.5, np.max(np.abs(x[:, 0]))+0.5)
h.fig.suptitle('Correlated Normal Distribution')
h.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95)
h.plot_marginals(sns.rugplot, color="#800000", height=.15, clip_on=True)
#h.savefig("NormalDistr.png")



norm = stats.norm([0],[1])
x_unif = norm.cdf(x)


# # plot correlated uniform ditribution, created from the
# # transformed normal distributions
h = sns.jointplot(x=x_unif[:, 0], y=x_unif[:, 1], kind='hex', gridsize=5)
h.ax_joint.plot(x_unif[:, 0], x_unif[:, 1], ".", color="#800000")
h.ax_marg_y.set_ylim(0, 1)
h.ax_marg_x.set_xlim(0, 1)
h.set_axis_labels('Y1', 'Y2', fontsize=16)
h.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95)
h.fig.suptitle('Correlated Uniform Distribution')

fig, ax1 = plt.subplots(ncols=2)
ax1[0].plot(x[:,0], x[:,1], 'x')
ax1[1].plot(x_unif[:,0], x_unif[:,1], 'x')


"""g = sns.jointplot(x=x[:, 0], y=x[:, 1])
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)"""

plt.show()
exit()

a, b = x_unif[:,0], x_unif[:,1]

nom_a_list = a
nom_b_list = b

diff = np.fabs(material_a_min - material_a_max)  # absolute val of each el
long_a_list = material_a_min + nom_a_list * diff  # pop with their real values
a_list = np.around(long_a_list, decimals=2)

diff = np.fabs(material_b_min - material_b_max)  # absolute val of each el
long_b_list = material_b_min + nom_b_list * diff  # pop with their real values
b_list = np.around(long_b_list, decimals=2)

#print("a_list", a_list)
#print("b_list", b_list)

fig, ax = plt.subplots()
fig2, ax_nom = plt.subplots()

ax.plot(a_list, b_list, 'x', label="a: mean=%.2f, std=%.2f\nb: mean=%.2f, std=%.2f" % (np.mean(a_list), np.std(a_list), np.mean(b_list), np.std(b_list)))
ax_nom.plot(a, b, 'x', label="a: mean=%.2f, std=%.2f\nb: mean=%.2f, std=%.2f" % (np.mean(a), np.std(a), np.mean(b), np.std(b)))


ax.set_title("Transformed data")
ax_nom.set_title("Nomalised data")

plt.show()

# fin
