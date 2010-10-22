#!/usr/bin/env ruby

if ARGV.length != 2
	exit
end

truth = Hash.new
total = 0

File.new(ARGV[0]).each_line do |line|
	args = line.split(" ")
	name = args[0].split(".")
	name = name[0].downcase
	if !truth[name]
		truth[name] = Array.new
	end
	truth[name].push({ :leftEye => { :x => args[1].to_f, :y => args[2].to_f },
					   :rightEye => { :x => args[3].to_f, :y => args[4].to_f },
					   :nose => { :x => args[5].to_f, :y => args[6].to_f },
					   :leftMouth => { :x => args[7].to_f, :y => args[8].to_f },
					   :centerMouth => { :x => args[9].to_f, :y => args[10].to_f },
					   :rightMouth => { :x => args[11].to_f, :y => args[12].to_f } })
	total += 1
end

fa = 0
tp = 0

File.new(ARGV[1]).each_line do |line|
	args = line.split(" ")
	name = args[0].split(".")
	name = name[0].downcase
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
			if (face[:leftEye][:x] > x && face[:leftEye][:x] < x + width && face[:leftEye][:y] > y && face[:leftEye][:y] < y + height &&
				face[:rightEye][:x] > x && face[:rightEye][:x] < x + width && face[:rightEye][:y] > y && face[:rightEye][:y] < y + height &&
				face[:nose][:x] > x && face[:nose][:x] < x + width && face[:nose][:y] > y && face[:nose][:y] < y + height &&
				face[:leftMouth][:x] > x && face[:leftMouth][:x] < x + width && face[:leftMouth][:y] > y && face[:leftMouth][:y] < y + height &&
				face[:centerMouth][:x] > x && face[:centerMouth][:x] < x + width && face[:centerMouth][:y] > y && face[:centerMouth][:y] < y + height &&
				face[:rightMouth][:x] > x && face[:rightMouth][:x] < x + width && face[:rightMouth][:y] > y && face[:rightMouth][:y] < y + height)
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
