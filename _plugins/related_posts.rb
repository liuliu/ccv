require 'jekyll/post'
require 'classifier'

module RelatedPosts

	# Used to remove #related_posts so that it can be overridden
	def self.included(klass)
		klass.class_eval do
			remove_method :related_posts
		end
	end

	# Calculate related posts.
	#
	# Returns [<Post>]
	def related_posts(posts)
		return [] unless posts.size > 1

		if self.content.size > 50
			self.class.lsi ||= begin
				puts "Running the classifier... this could take a while."
				lsi = Classifier::LSI.new({:auto_rebuild => false})
				posts.each do |x|
					$stdout.print("+")
					$stdout.flush
					lsi.add_item(x)
				end
				puts "."
				puts "building index..."
				lsi.build_index
				puts "done building index"
				lsi
			end

			related = self.class.lsi.find_related(self.content, 6)
			related - [self]
		else
			highest_freq = Jekyll::Post.category_freq(posts).values.max
			related_scores = Hash.new(0)
			posts.each do |post|
				post.categories.each do |category|
					if self.categories.include?(category) && post != self
						cat_freq = Jekyll::Post.category_freq(posts)[category]
						related_scores[post] += (1 + highest_freq-cat_freq)
					end
				end
			end

			Jekyll::Post.sort_related_posts(related_scores)[0..6] - [self]
		end
	end

	module ClassMethods
		# Calculate the frequency of each category.
		#
		# Returns {category => freq, category => freq, ...}
		def category_freq(posts)
			return @category_freq if @category_freq
			@category_freq = Hash.new(0)
			posts.each do |post|
				post.categories.each {|category| @category_freq[category] += 1}
			end
			@category_freq
		end

		# Sort the related posts in order of their score and date
		# and return just the posts
		def sort_related_posts(related_scores)
			related_scores.sort do |a,b|
				if a[1] < b[1]
					1
				elsif a[1] > b[1]
					-1
				else
					b[0].date <=> a[0].date
				end
			end.collect {|post,freq| post}
		end
	end

end

module Jekyll
	class Post
		include RelatedPosts
		extend RelatedPosts::ClassMethods
	end
end
