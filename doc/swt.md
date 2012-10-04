SWT: Stroke Width Transform
===========================

*(This documentation is still largely work in progress, use with caution)*

What's SWT?
-----------

The original paper refers to: Stroke Width Transform, Boris Epshtein, Yonathan Wexler,
and Eyal Ofek 2010.

How it works?
-------------

It is a long story, as always, please read their paper. SWT tries to capture the
only text effective features and using geometric signature of text to filter out
non-text areas. As a result, SWT gives you reliable text regions that is language
neutral. Try it yourself:

	./swtdetect <Your Image contains Text> | ./swtdraw.rb <Your Image contains Text> output.png

Checkout output.png, luckily, the text area is labeled.

What about performance?
-----------------------

SWT is quite fast. The SWT without scale-invariant support (multi-scale) can run
on a 640x480 photo for well under 50 milliseconds on my laptop. By extending SWT
to multi-scale, the accuracy increased by about 10% with about 2~4 times longer
running time.

Accuracy-wise:

ccv's SWT implementation performs on ICDAR 2003 dataset achieved similar performance
with what Epshtein et al. reported in their paper, namely, with the old measure
method described in ICDAR 2003 contest, ccv's implementation was able to achieve
precision rate at 66% and recall rate at 59% (numbers reported in the paper are
precision rate 73% and recall rate at 60%).

However, these results are quite out-dated, and by using [ICDAR 2011 dataset](http://robustreading.opendfki.de/wiki/SceneText),
more meaningful comparison is possible.

With ccv's scale-invariant SWT implementation, and do parameter search on ICDAR
2011's training dataset, I was able to achieve:

	precision: 59%
	recall: 61%
	harmonic mean: 60%

Which would rank around 2nd to 3rd place in the chart. Please note that other
methods in comparison are language specific, thus, were trained with additional
character shape information using SVM or Adaboost where as SWT is language neutral
and doesn't use any language specific features.

Speed-wise:

How can I adopt SWT for my application?
---------------------------------------