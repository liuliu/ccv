#!/usr/bin/env ruby

exit unless ARGV.length == 2

truth = Hash.new
file = File.new(ARGV[0])
filename = nil
file.each do |line|
	if line =~ /\d+\s\d+\s\d+\s\d+/
		truth[filename] = Array.new unless truth.has_key? filename
		nb = line.split " "
		truth[filename] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	else
		filename = line
	end
end

estimate = Hash.new
file = File.new(ARGV[1])
file.each do |line|
	if line =~ /\d+\s\d+\s\d+\s\d+/
		estimate[filename] = Array.new unless estimate.has_key? filename
		nb = line.split(" ")
		estimate[filename] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	else
		filename = line
	end
end

total = 0
precision = 0
estimate.each do |fn, rects|
	rects.each do |rect|
		match = 0
		next unless estimate.has_key? fn
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
		next unless estimate.has_key? fn
		estimate[fn].each do |target|
			match = [[[target[:x] + target[:width], rect[:x] + rect[:width]].min - [target[:x], rect[:x]].max, 0].max * [[target[:y] + target[:height], rect[:y] + rect[:height]].min - [target[:y], rect[:y]].max, 0].max, match].max
		end
		recall += match.to_f / (rect[:width] * rect[:height]).to_f
		total += 1
	end
end

print "recall: " + (((recall.to_f / total.to_f) * 10000).round / 100).to_s + "%\n"
