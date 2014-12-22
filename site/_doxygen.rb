#!/usr/bin/env ruby

require 'nokogiri'

exit unless ARGV.length == 2

def markdown_safe x
	return x.gsub('_', '\_').gsub('|', '\|')
end

def remove_ulinks doc
	# replace all ulink to be text
	doc.search('ulink').each do |ulink|
		ulink.replace Nokogiri::XML::Text.new(ulink.content, doc)
	end
end


def merge_structs structs, para
	structs_re = /[\w]+\_t/
	matches = para.match structs_re
	return unless matches != nil
	structs.merge matches.to_a
end

def output_function file, function
	# tries to print needed structs just down the functions, therefore, we log which structs appeared in the desc, which is in the form xxx_t
	structs = Set.new
	paras = function.xpath './detaileddescription/para'
	return structs if paras.length == 0
	desc = Array.new
	paras.each do |para|
		para = para.at('./text()').content.strip
		merge_structs structs, para
		desc << para if para.length > 0
	end
	return structs if desc.length == 0
	name = function.at('./name').content
	print ' - Generating function ' + name + "\n"
	name = markdown_safe name
	file << "\n" + name + "\n" + ('-' * name.length) + "\n\n"
	case function['kind']
		when 'function'
			proto = function.at('./definition').content + function.at('./argsstring').content
		when 'define'
			proto = function.at('./name').content
			defnames = Array.new
			function.xpath('./param/defname').each do |defname|
				defnames << defname.content.strip
			end
			proto = proto + '(' + defnames.join(', ') + ')' if defnames.length > 0
	end
	file << "\t" + proto + "\n\n" + markdown_safe(desc.join("\n\n")) + "\n"
	params = function.xpath "./detaileddescription/para/parameterlist[@kind='param']/parameteritem"
	file << "\n" if params.length > 0
	params.each do |param|
		paramnames = Array.new
		param.xpath('./parameternamelist/parametername').each do |paramname|
			paramnames << markdown_safe(paramname.content.strip)
		end
		file << " * **" << paramnames.join(", ") << "**: "
		desc = param.at('./parameterdescription/para').content.strip
		merge_structs structs, desc
		file << markdown_safe(desc) << "\n"
	end
	retdesc = function.at "./detaileddescription/para/simplesect[@kind='return']/para"
	if retdesc != nil
		retdesc = retdesc.content.strip
		merge_structs structs, retdesc
		file << "\n**return**: " + markdown_safe(retdesc) + "\n" 
	end
	return structs
end

def alt_name desc
	alt_name_re = /^\[[\w\.]+\]/
	matches = desc.match alt_name_re
	return nil unless matches != nil
	return matches[0][1, matches[0].length - 2]
end

def output_struct file, structname, doc_group
	structs = Set.new
	compoundname = doc_group.at('./compoundname').content.strip
	return structs unless compoundname == structname
	sections = doc_group.xpath './sectiondef'
	first_struct = true
	sections.each do |section|
		variables = section.xpath "./memberdef[@kind='variable']"
		available_desc = false
		variables.each do |variable|
			para = variable.at './detaileddescription/para'
			available_desc = true if para != nil
			break if available_desc
		end
		# next section if no available desc anywhere
		next if !available_desc
		if first_struct
			# if we haven't print the name of the struct yet, print it now
			print ' - Generating struct ' + compoundname + "\n"
			compoundname = markdown_safe compoundname
			file << "\n" + compoundname + "\n" + ('-' * compoundname.length) + "\n\n"
			header = section.at './header'
			file << header.content.strip + "\n\n" if header != nil
			first_struct = false
		else
			header = section.at './header'
			file << "\n" + header.content.strip + "\n\n" if header != nil
		end
		vars = Hash.new
		variables.each do |variable|
			paras = variable.xpath './detaileddescription/para'
			next if paras.length == 0
			paras.each do |para|
				desc = para.content.strip
				alt_name = alt_name desc
				desc = desc.sub('[' + alt_name + ']', '').strip if alt_name != nil
				merge_structs structs, desc
				name =
					if alt_name != nil
						markdown_safe alt_name.strip
					else
						markdown_safe variable.at('./name').content
					end
				desc = markdown_safe desc
				vars[name] = desc if !vars.has_key?(name)
			end
		end
		vars_a = Array.new
		vars.each do |name, desc|
			vars_a << ' * **' + name + '**: ' + desc
		end
		file << vars_a.sort.join("\n") + "\n"
	end
	return structs
end

def open_and_output_struct out_structs, file, structname, doc_group, dirname
	doc_group.xpath('./innerclass').each do |innerclass|
		if innerclass.content.strip == structname
			doc = Nokogiri::XML(open(dirname + '/' + innerclass['refid'] + '.xml'))
			remove_ulinks doc
			structs = output_struct file, structname, doc.at('./doxygen/compounddef')
			structs = structs - out_structs
			out_structs.merge structs
			structs.each do |struct|
				open_and_output_struct out_structs, file, struct, doc_group, dirname
			end
		end
	end
end

require 'pathname'
require 'set'

dirname = Pathname.new(ARGV[0]).dirname.to_s
outdir = Pathname.new(ARGV[1])
exit unless outdir.directory?
outdir = outdir.to_s

doc = Nokogiri::XML(open ARGV[0])

remove_ulinks doc

doc_group = doc.at './doxygen/compounddef'

compoundname = doc_group.at './compoundname'

desc = doc_group.at './title'

slug = compoundname.content.gsub('_', '-')

title = 'lib/' + compoundname.content + '.c'

print 'Generating for ' + title + "\n"

filename = '0000-01-01-' + slug + '.markdown'

file = File.open(outdir + "/" + filename, 'w+')

file << "---\nlayout: page\nlib: ccv\nslug: " + slug + "\nstatus: publish\ntitle: " + title + "\ndesc: " + desc + "\ncategories:\n- lib\n---\n"

para = doc_group.at './detaileddescription/para'

file << "\n" + para.content.strip.capitalize + "\n" if para != nil

functions = doc_group.xpath ".//memberdef[@kind='function'] | .//memberdef[@kind='define']"
out_structs = Set.new
functions.each do |function|
	structs = output_function file, function
	structs = structs - out_structs
	out_structs.merge structs
	structs.each do |struct|
		open_and_output_struct out_structs, file, struct, doc_group, dirname
	end
end

print 'Done for ' + title + "\n"
