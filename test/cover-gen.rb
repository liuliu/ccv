#!/usr/bin/env ruby

require 'pathname'
require 'date'
require 'fileutils'

abort "cover-gen.rb OUTPUT\n" unless ARGV.length == 1

output_dir = ARGV[0] + '/' + DateTime.now.strftime(format='%F-%H%M%S')

Dir.glob("**/*.html").each do |fn|
	file = Pathname.new fn
	target_dir = output_dir + '/' + file.dirname.to_s
	FileUtils::mkdir_p target_dir unless File.exists? target_dir
	FileUtils::mv(file.realpath.to_s, target_dir)
end
