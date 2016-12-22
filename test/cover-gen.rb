#!/usr/bin/env ruby

require 'pathname'
require 'date'
require 'fileutils'

abort "cover-gen.rb OUTPUT\n" unless ARGV.length == 1

output_dir = Pathname.new(ARGV[0]) + DateTime.now.strftime(format='%F-%H%M%S')

Dir.glob("**/*.profraw").each do |fn|
	file = Pathname.new fn
	%x[llvm-profdata merge -sparse #{fn} -o #{fn}.profdata]
	relative_dir = file.dirname
	filename = file.basename '.profraw'
	%x[llvm-cov show ./#{relative_dir + filename} -instr-profile #{fn}.profdata -format=html -show-line-counts-or-regions -output-dir="#{output_dir + relative_dir + filename}"]
end

Dir.glob(output_dir + "**/*.html").each do |fn|
	%x[sed -i "s/pre {/pre {\\n  -moz-tab-size\: 4;\\n  tab-size\: 4;/" "#{fn}"]
end
