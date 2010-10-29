<?php

	$url = urldecode(filter_var($_GET['src']));
	echo file_get_contents($url);
