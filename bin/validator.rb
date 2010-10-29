#!/usr/bin/env ruby

exit unless ARGV.length == 2

truth = Hash.new
total = 0

File.new(ARGV[0]).each_line do |line|
	args = line.split(" ")
	name = args[0].split(".")[0].downcase
	truth[name] = Array.new if truth[name].nil?
	truth[name] << {:left_eye => {:x => args[1].to_f, :y => args[2].to_f},
					:right_eye => {:x => args[3].to_f, :y => args[4].to_f},
					:nose => {:x => args[5].to_f, :y => args[6].to_f},
					:left_mouth => {:x => args[7].to_f, :y => args[8].to_f},
					:center_mouth => {:x => args[9].to_f, :y => args[10].to_f},
					:right_mouth => {:x => args[11].to_f, :y => args[12].to_f}}
	total += 1
end

fa = 0
tp = 0

File.new(ARGV[1]).each_line do |line|
	args = line.split(" ")
	name = args[0].split(".")[0].downcase
	if !truth[name]
		fa += 1
	else
		# relaxing the box
		x = args[1].to_f
		y = args[2].to_f
		width = args[3].to_f
		height = args[4].to_f
		x -= width * 0.25
		y -= height * 0.25
		width *= 1.5
		height *= 1.5
		outlier = true
		truth[name].each do |face|
			if (face[:left_eye][:x] > x && face[:left_eye][:x] < x + width &&
				face[:left_eye][:y] > y && face[:left_eye][:y] < y + height &&
				face[:right_eye][:x] > x && face[:right_eye][:x] < x + width &&
				face[:right_eye][:y] > y && face[:right_eye][:y] < y + height &&
				face[:nose][:x] > x && face[:nose][:x] < x + width &&
				face[:nose][:y] > y && face[:nose][:y] < y + height &&
				face[:left_mouth][:x] > x && face[:left_mouth][:x] < x + width &&
				face[:left_mouth][:y] > y && face[:left_mouth][:y] < y + height &&
				face[:center_mouth][:x] > x && face[:center_mouth][:x] < x + width &&
				face[:center_mouth][:y] > y && face[:center_mouth][:y] < y + height &&
				face[:right_mouth][:x] > x && face[:right_mouth][:x] < x + width &&
				face[:right_mouth][:y] > y && face[:right_mouth][:y] < y + height)
				outlier = false
				break
			end
		end
		if outlier
			fa += 1
		else
			tp += 1
		end
	end
end

print ((tp.to_f / total.to_f * 10000).round / 100.0).to_s + "% ("+ fa.to_s + ")\n"
