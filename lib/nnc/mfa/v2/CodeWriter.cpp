#include "CodeWriter.hpp"

#include <assert.h>

void CodeWriter::operator+=(std::string text) {
  if (!ignore_ident_ && !text.empty()) AppendIdent(stream_);

  while (true) {
    auto begin = text.find("{{");
    if (begin == std::string::npos) { break; }

    auto end = text.find("}}");
    if (end == std::string::npos || end < begin) { break; }

    // Write all the text before the first {{ into the stream.
    stream_.write(text.c_str(), begin);

    // The key is between the {{ and }}.
    const std::string key = text.substr(begin + 2, end - begin - 2);

    // Find the value associated with the key.  If it exists, write the
    // value into the stream, otherwise write the key itself into the stream.
    auto iter = value_map_.find(key);
    if (iter != value_map_.end()) {
      const std::string &value = iter->second;
      stream_ << value;
    } else {
      assert(false && "could not find key");
      stream_ << key;
    }

    // Update the text to everything after the }}.
    text = text.substr(end + 2);
  }
  if (!text.empty() && text.back() == '\\') {
    text.pop_back();
    ignore_ident_ = true;
    stream_ << text;
  } else {
    ignore_ident_ = false;
    stream_ << text << std::endl;
  }
}

void CodeWriter::AppendIdent(std::stringstream &stream) {
  int lvl = cur_ident_lvl_;
  while (lvl--) {
    stream.write(pad_.c_str(), static_cast<std::streamsize>(pad_.size()));
  }
}

