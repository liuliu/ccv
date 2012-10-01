#!/usr/bin/env ruby

# to make a better roc graph, I uses what DetEval do for precision and recall computation

exit unless ARGV.length == 2

one_g = 0.8
one_d = 0.4
om_one = 0.8
center_diff_thr = 1.0

truth = Hash.new
file = File.new ARGV[0]
filename = nil
file.each do |line|
	if line =~ /-{0,1}\d+\s-{0,1}\d+\s-{0,1}\d+\s-{0,1}\d+/
		truth[filename] = Array.new unless truth.has_key? filename
		nb = line.split " "
		truth[filename] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	else
		filename = line.strip
	end
end

estimate = Hash.new
file = File.new ARGV[1]
file.each do |line|
	if line =~ /-{0,1}\d+\s-{0,1}\d+\s-{0,1}\d+\s-{0,1}\d+/
		estimate[filename] = Array.new unless estimate.has_key? filename
		nb = line.split(" ")
		estimate[filename] << { :x => nb[0].to_f, :y => nb[1].to_f, :width => nb[2].to_f, :height => nb[3].to_f }
	else
		filename = line.strip
	end
end

recall = 0
precision = 0
truth.each do |fn, rects|
	next unless estimate.has_key? fn
	cG = Array.new rects.count, 0
	cD = Array.new estimate[fn].count, 0
	mG = Array.new rects.count do Array.new estimate[fn].count, 0 end
	mD = Array.new estimate[fn].count do Array.new rects.count, 0 end
	rects.each_index do |i|
		rect = rects[i]
		estimate[fn].each_index do |j|
			target = estimate[fn][j]
			match = [[target[:x] + target[:width], rect[:x] + rect[:width]].min - [target[:x], rect[:x]].max, 0].max * [[target[:y] + target[:height], rect[:y] + rect[:height]].min - [target[:y], rect[:y]].max, 0].max
			if match > 0.0001
				mG[i][j] = match / (rect[:width] * rect[:height])
				mD[j][i] = match / (target[:width] * target[:height])
				cG[i] += 1
				cD[j] += 1
			end
		end
	end
	tG = Array.new rects.count, false
	tD = Array.new estimate[fn].count, false
	# one to one match
	rects.each_index do |i|
		rect = rects[i]
		next if cG[i] != 1
		estimate[fn].each_index do |j|
			target = estimate[fn][j]
			next if cD[j] != 1
			if mG[i][j] >= one_g and mD[j][i] >= one_d
				dx = (target[:x] + target[:width] * 0.5) - (rect[:x] + rect[:width] * 0.5)
				dy = (target[:y] + target[:height] * 0.5) - (rect[:y] + rect[:height] * 0.5)
				d = Math.sqrt(dx**2 + dy**2) * 2.0 / (Math.sqrt(target[:width]**2 + target[:height]**2) + Math.sqrt(rect[:width]**2 + rect[:height]**2))
				if d < center_diff_thr
					recall += 1.0
					precision += 1.0
					tG[i] = tD[j] = true
				end
			end
		end
	end
	# one to many match, starts with ground truth
	rects.each_index do |i|
		next if tG[i] or cG[i] <= 1
		one_sum = 0
		many = Array.new
		estimate[fn].each_index do |j|
			next if tD[j]
			many_single = mD[j][i]
			if many_single >= one_d
				one_sum += mG[i][j]
				many << j
			end
		end
		if many.count == 1
			# only one qualified, degrade to one to one match
			if mG[i][many[0]] >= one_g and mD[many[0]][i] >= one_d
				recall += 1.0
				precision += 1.0
				tG[i] = tD[many[0]] = true
			end
		elsif one_sum >= one_g
			many.each do |j|
				tD[j] = true
			end
			recall += om_one
			precision += om_one / (1.0 + Math.log(many.count))
		end
	end
	# one to many match, with estimate
	estimate[fn].each_index do |j|
		next if tD[j] or cD[j] <= 1
		one_sum = 0
		many = Array.new
		rects.each_index do |i|
			next if tG[i]
			many_single = mG[i][j]
			if many_single >= one_g
				one_sum += mD[j][i]
				many << i
			end
		end
		if many.count == 1
			# only one qualified, degrade to one to one match
			if mG[many[0]][j] >= one_g and mD[j][many[0]] >= one_d
				recall += 1.0
				precision += 1.0
				tG[many[0]] = tD[j] = true
			end
		elsif one_sum >= one_d
			many.each do |i|
				tG[i] = true
			end
			recall += om_one / (1.0 + Math.log(many.count))
			precision += om_one
		end
	end
end

total_estimate = 0
estimate.each do |fn, rects|
	total_estimate += rects.count
end
precision = precision.to_f / total_estimate.to_f

total_truth = 0
truth.each do |fn, rects|
	total_truth += rects.count
end
recall = recall.to_f / total_truth.to_f

print "precision: " + ((precision * 10000).round / 100).to_s + "%\n"
print "recall: " + ((recall * 10000).round / 100).to_s + "%\n"
print "harmonic mean: " + ((2.0 * precision * recall / (precision + recall) * 10000).round / 100).to_s + "%\n"
