#!/usr/bin/env ruby

require 'nokogiri'

exit unless ARGV.length == 1

def markdown_safe x
	return x.gsub('_', '\_').gsub('|', '\|')
end

doc = Nokogiri::XML(open ARGV[0])

# replace all ulink to be text
doc.search('ulink').each do |ulink|
	ulink.replace Nokogiri::XML::Text.new(ulink.content, doc)
end

doc_group = doc.at './doxygen/compounddef'

compoundname = doc_group.at './compoundname'

desc = doc_group.at './title'

slug = compoundname.content.gsub('_', '-')

title = 'lib/' + compoundname.content + '.c'

filename = '0000-01-01-' + slug + '.markdown'

file = File.open filename, 'w+'

file << "---\nlayout: page\nlib: ccv\nslug: " + slug + "\nstatus: publish\ntitle: " + title + "\ndesc: " + desc + "\ncategories:\n- lib\n---\n"

functions = doc_group.xpath ".//memberdef[@kind='function']"

functions.each do |function|
	paras = function.xpath './detaileddescription/para'
	next if paras.length == 0
	desc = Array.new
	paras.each do |para|
		para = para.at('./text()').content.strip
		desc << para if para.length > 0
	end
	next if desc.length == 0
	name = markdown_safe function.at('./name').content
	file << "\n" + name + "\n" + ('-' * name.length) + "\n\n"
	proto = function.at('./definition').content + function.at('./argsstring').content
	file << "\t" + proto + "\n\n" + desc.join("\n\n") + "\n"
	params = function.xpath "./detaileddescription/para/parameterlist[@kind='param']/parameteritem"
	file << "\n" if params.length > 0
	params.each do |param|
		paramnames = Array.new
		param.xpath('./parameternamelist/parametername').each do |paramname|
			paramnames << markdown_safe(paramname.content.strip)
		end
		file << " * **" << paramnames.join(", ") << "**: "
		desc = param.at './parameterdescription/para'
		file << markdown_safe(desc.content) << "\n"
	end
	retdesc = function.at "./detaileddescription/para/simplesect[@kind='return']/para"
	file << "\n**return**: " + retdesc.content.strip + "\n" if retdesc != nil
end
