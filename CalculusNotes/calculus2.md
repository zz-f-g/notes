# Calculus2: unlimited and limited integrals

## Concepts

---

## Exercise

### 第一类换元法

---

I

$$
\begin{aligned}
\int \frac{x+1}{x(1 + x e^x)} \mathrm{d}x &= \int \frac{(x+1)e^x}{x e^x (1+xe^x)} \mathrm{d}x \\
&= \int \frac{\mathrm{d}(x e^x)}{x e^x} - \int \frac{\mathrm{d}(x e^x)}{1 + x e^x} \\
&= \ln \left| \frac{x e^x}{1 + x e^x}\right|
\end{aligned}
$$

***Attention***:
$$
(x e^x)' = (x+1)e^x
$$

---

II

$$
\begin{aligned}
\int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{\mathrm{d}x}{(1- \cos x)\sin^2 x} &= \int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{(1 + \cos x)\mathrm{d} x}{\sin^4 x} \\
&= \int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{{\mathrm{d}x}}{\sin^4 x} + \int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{\mathrm{d}(\sin x)}{\sin^4 x} \\
&= - \int_{\frac{\pi}{2}}^{2 \arctan 2} (1 + \cot^2 x) \mathrm{d}(\cot x) - \left.\frac{1}{3 \sin^3 x}\right|_{\frac{\pi}{2}}^{2 \arctan 2} \\
&= \left.\left(- \cot x - \frac{\cot^3 x}{3} - \frac{1}{3 \sin^3 x}\right)\right|_{\frac{\pi}{2}}^{2 \arctan 2} \\
&= \frac{55}{96}
\end{aligned}
$$

OR
$$
\begin{aligned}
\int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{\mathrm{d}x}{(1-\cos x)\sin^2 x} &= - \int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{\mathrm{d}(\cot x)}{1 - \frac{\cot x}{\sqrt{1 + \cot^2 x}}} \\
&= - \int_{\frac{\pi}{2}}^{2 \arctan 2} \frac{\sqrt{1 + \cot^2 x}\mathrm{d} x}{\sqrt{1 + \cot^2 x}-\cot x} \\
&= - \int_{0}^{-\frac{3}{4}} (1 + u^2 + \sqrt{1 + u^2} u) \mathrm{d}u \\
&= - \left.\left(u + \frac{u^3}{3} + \frac{1}{3} (1+u^2)^{\frac{3}{2}}\right)\right|_{0}^{-\frac{3}{4}}
\end{aligned}
$$

***Attention***:
也可以用万能公式换元。

---

III

$$
\begin{aligned}
\int_0^{\frac{\pi}{2}} \frac{\mathrm{d}x}{1 + \tan^{\sqrt{2}}x} & \overset{u = \frac{\pi}{2} - x}{=} \int_{\pi/2}^0 \frac{- \mathrm{d}u}{1 + \tan^{\sqrt{2}}(\frac{\pi}{2} - u)} \\
&= \int_0^{\pi /2} \frac{\mathrm{d}u}{1 + \frac{1}{\tan^{\sqrt{2}}u}} \\
&= \int_0^{\pi /2} \frac{\tan^{\sqrt{2}}u \mathrm{d}u}{1 + {\tan^{\sqrt{2}}u}} \\
&\overset{x=u}{=} \int_0^{\pi /2} \frac{\tan^{\sqrt{2}}x \mathrm{d}x}{1 + {\tan^{\sqrt{2}}x}} \\
\int_0^{\frac{\pi}{2}} \frac{\mathrm{d}x}{1 + \tan^{\sqrt{2}}x} & = \int_0^{\pi /2} \frac{\tan^{\sqrt{2}}x \mathrm{d}x}{1 + {\tan^{\sqrt{2}}x}} = \frac{1}{2} \int_0^{\frac{\pi}{2}} \mathrm{d} x = \frac{\pi}{4}
\end{aligned}
$$

***Attention***:
遇到不能写出原函数的定积分，需要转化成和原来积分有关系的形式。

---

IV

?There are some questions in following process.
$$
\begin{aligned}
\lim_{x \rightarrow 0} \frac{\int_0^{x^2}t^2 \sin \sqrt{x^2 - t^2} \mathrm{d} t}{x^7} &= \lim_{x \rightarrow 0} \frac{x^4 \sin \sqrt{x^2 - x^4} \cdot 2x}{7 x^6} \\
&= \frac{2}{7} \lim_{x \rightarrow 0} \frac{\sin x \sqrt{1 - x^2}}{x} \\
&= \frac{2}{7}
\end{aligned}
$$

因为上面的被积函数中显含有 $x$，不可以直接讲上下限作为自变量进行链式求导。

---

V

$$
\begin{aligned}
\int_0^{\pi/2} \frac{\mathrm{d}x}{1 + \cos \theta \cos x} &= \int_0^{\pi/2} \frac{\mathrm{d}x}{1 + \cos \theta \frac{1 - \tan^2 \frac{x}{2}}{1 + \tan^2 \frac{x}{2}}} \\
&= \int_0^{\pi/2} \frac{(1+\tan^2 \frac{x}{2})\mathrm{d}x}{1+\tan^2 \frac{x}{2}+ \cos \theta (1 - \tan^2 \frac{x}{2})} \\
&= 2 \int_0^{\pi/2} \frac{\mathrm{d}(\tan \frac{x}{2})}{1 + \cos \theta + (1 - \cos \theta) \tan^2 \frac{x}{2}} \\
&= 2 \int_0^1 \frac{\mathrm{d}x}{1+\cos \theta + (1 - \cos \theta)x^2} \\
&= \frac{2}{1 + \cos \theta} \int_0^1 \frac{\mathrm{d}x}{1 + \left(\sqrt{\frac{1 - \cos \theta}{1 + \cos \theta}}x\right)^2} \\
&= 2 \sqrt{\frac{1}{(1 - \cos \theta)(1 + \cos \theta)}} \int_0^{\sqrt{\frac{1 - \cos \theta}{1 + \cos \theta}}} \frac{\mathrm{d}x}{1 + x^2} \\
&= \frac{2}{|\sin \theta|} \arctan \sqrt{\frac{1 - \cos \theta}{1 + \cos \theta}} \\
&= \frac{2}{|\sin \theta|} \arctan \left| \tan \frac{\theta}{2} \right| \\
&= \frac{\theta}{\sin \theta} & (\theta \in (0, \pi))
\end{aligned}
$$

---

VI

$$
\begin{aligned}
\int \frac{\sec x \tan^2 x + \sec^3 x}{\sqrt{\sec^4 x + \tan^4 x}} \mathrm{d}x &= \int \frac{\tan x \mathrm{d} \sec x + \sec x \mathrm{d} \tan x}{\sqrt{(\sec^2 x - \tan^2 x)^2 + 2 \sec^2 x \tan^2 x}} \\
&= \int \frac{\mathrm{d} \sec x \tan x}{\sqrt{1 + 2 (\sec x \tan x)^2}} \\
&= \frac{\sqrt{2}}{2} \sinh^{-1} (\sqrt{2} \sec x \tan x) + C
\end{aligned}
$$

This is an end for this document. I write this in order to prevent the preview interface to be gone.

Here I want to say something to my self:

This is a notebook that seems to be foolish. As a matter of fact, before this foolish notebook, you were always doing something foolish and this is far from the first time. All this foolish things makes you a people different-or more narcissisticly, distinguishing.

This world need more people with a foolish mind instead of smart or sharp ones. The former can't be gained until you devote matched vigor and time absorbed.
