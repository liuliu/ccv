{
  "targets": [{
    "target_name": "ccv_node",
    "sources": [
      "ccv_node.cc",
      "bbf_node.cc"
    ],
    'include_dirs': [
      '../lib'
    ],
    'link_settings': {
      'libraries': [
        '-L../../lib -lccv'
      ],
    },
  }]
}
