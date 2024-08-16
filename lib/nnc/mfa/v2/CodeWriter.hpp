#ifndef MFA_CODE_WRITER_HPP_
#define MFA_CODE_WRITER_HPP_

#include <map>
#include <sstream>

class CodeWriter {
 public:
  CodeWriter(std::string pad = std::string())
      : pad_(pad), cur_ident_lvl_(0), ignore_ident_(false) {}

  // Clears the current "written" code.
  void Clear() {
    stream_.str("");
    stream_.clear();
  }

  // Associates a key with a value.  All subsequent calls to operator+=, where
  // the specified key is contained in {{ and }} delimiters will be replaced by
  // the given value.
  void SetValue(const std::string &key, const std::string &value) {
    value_map_[key] = value;
  }

  std::string GetValue(const std::string &key) const {
    const auto it = value_map_.find(key);
    return it == value_map_.end() ? "" : it->second;
  }

  // Appends the given text to the generated code as well as a newline
  // character.  Any text within {{ and }} delimiters is replaced by values
  // previously stored in the CodeWriter by calling SetValue above.  The newline
  // will be suppressed if the text ends with the \\ character.
  void operator+=(std::string text);

  // Returns the current contents of the CodeWriter as a std::string.
  std::string ToString() const { return stream_.str(); }

  // Increase ident level for writing code
  void IncrementIdentLevel() { cur_ident_lvl_++; }
  // Decrease ident level for writing code
  void DecrementIdentLevel() {
    if (cur_ident_lvl_) cur_ident_lvl_--;
  }

  void SetPadding(const std::string &padding) { pad_ = padding; }

 private:
  std::map<std::string, std::string> value_map_;
  std::stringstream stream_;
  std::string pad_;
  int cur_ident_lvl_;
  bool ignore_ident_;

  // Add ident padding (tab or space) based on ident level
  void AppendIdent(std::stringstream &stream);
};

#endif
