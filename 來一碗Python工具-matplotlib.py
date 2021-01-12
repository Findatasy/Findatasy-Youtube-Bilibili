# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 23:46:06 2021

@author: Findatasy
"""

import matplotlib.pyplot as plt

import matplotlib
#防止中文亂碼
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

import numpy as np

#%%

# =============================================================================
# generate data and simple plot
# =============================================================================
np.random.normal(loc=100, scale=1, size=20)

stock_A = np.random.normal(loc=100, scale=1, size=20)
stock_B = np.random.normal(loc=80, scale=1, size=20)
time = list(range(1,21))


plt.plot(time, stock_A)
plt.plot(time, stock_B)


# =============================================================================
# navigate between figures
# =============================================================================

plt.figure(0)
plt.plot(time, stock_A)
plt.figure(1)
plt.plot(time, stock_B)

# go back to figure 0
plt.figure(0)
plt.plot(time, stock_B)


# =============================================================================
# different types of plot
# =============================================================================

plt.plot(time, stock_A)
plt.plot(time, stock_A, 'k--')
plt.bar(time, stock_A)
plt.scatter(time, stock_A)
plt.pie(stock_A)
plt.polar(stock_A)

# draw a heart
T = np.linspace(0, 2 * np.pi, 1024)  # Angle range 0-2*pi, divided into 1024 equal parts
plt.axes(polar=True)    # Turn on polar coordinate mode
plt.plot(T, 1. - np.sin(T), color="r")


# =============================================================================
# Misc
# =============================================================================

plt.xlabel("I am X")
plt.ylabel("I am Y")
plt.title("中文标题无乱码喔！")
# remove margins
plt.tight_layout()

# =============================================================================
# create subplots
# =============================================================================

# M1: create using plt.figure()
# then plot each one use fig1.add_subplot()
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,2,1) # 2*2=4 figures, select the 1st
ax1.plot(time, stock_A)

# M2: create using plt.subplots()
fig2, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].plot(time, stock_A)
axes[0,1].plot(time, stock_B, color='orange')

# color ways
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html