<?php

	$url = urldecode(filter_var($_GET['src']));
	if (strtolower(substr($url, 0, 7)) == 'http://' || strtolower(substr($url, 0, 8) == 'https://'))
	{
		$path = '/tmp/'.basename($url);
		$output = fopen($path, 'w+');
		$curl = curl_init($url);
		$options = array(CURLOPT_HEADER => false,
						 CURLOPT_TIMEOUT => 30,
						 CURLOPT_FILE => $output,
						 CURLOPT_FOLLOWLOCATION => true);
		curl_setopt_array($curl, $options);
		$result = curl_exec($curl);
		$content_type = curl_getinfo($curl, CURLINFO_CONTENT_TYPE);
		curl_close($curl);
		fclose($output);
		header('Content-Type: '.$content_type);
		header('Content-Length: '.filesize($path));
		readfile($path);
	}
