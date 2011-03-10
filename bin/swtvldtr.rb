#!/usr/bin/env ruby

exit unless ARGV.length == 2

truth = Hash.new

file = File.new(ARGV[0])
images = file.readline.to_i
for i in 1..images do
	fn = file.readline
	locations = file.readline.to_i
	truth[fn] = Array.new
	for j in 1..locations do
		nb = file.readline.split(" ")
		truth[fn] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	end
end

estimate = Hash.new

file = File.new(ARGV[1])
images = file.readline.to_i
for i in 1..images do
	fn = file.readline
	locations = file.readline.to_i
	estimate[fn] = Array.new
	for j in 1..locations do
		nb = file.readline.split(" ")
		estimate[fn] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	end
end

total = 0
precision = 0
estimate.each do |fn, rects|
	rects.each do |rect|
		match = 0
		truth[fn].each do |target|
			match = [[[target[:x] + target[:width], rect[:x] + rect[:width]].min - [target[:x], rect[:x]].max, 0].max * [[target[:y] + target[:height], rect[:y] + rect[:height]].min - [target[:y], rect[:y]].max, 0].max, match].max
		end
		precision += match.to_f / (rect[:width] * rect[:height]).to_f
		total += 1
	end
end

print "precision: " + (((precision.to_f / total.to_f) * 10000).round / 100).to_s + "%\n"

total = 0
recall = 0
truth.each do |fn, rects|
	rects.each do |rect|
		match = 0
		estimate[fn].each do |target|
			match = [[[target[:x] + target[:width], rect[:x] + rect[:width]].min - [target[:x], rect[:x]].max, 0].max * [[target[:y] + target[:height], rect[:y] + rect[:height]].min - [target[:y], rect[:y]].max, 0].max, match].max
		end
		recall += match.to_f / (rect[:width] * rect[:height]).to_f
		total += 1
	end
end

print "recall: " + (((recall.to_f / total.to_f) * 10000).round / 100).to_s + "%\n"
