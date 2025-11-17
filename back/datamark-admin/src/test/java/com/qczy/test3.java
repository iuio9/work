package com.qczy;

import com.qczy.utils.FileFormatSizeUtils;
import com.qczy.utils.URLUtils;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class test3 {
    public static void main(String[] args) throws UnsupportedEncodingException {

        // 第一个文件名
        String originalFileName = "500kV安蔡一线#0087塔-0086塔-间隔棒大到小中相第2个间隔棒销钉未打开4-DJI-2900.jpg";
        System.out.println(URLUtils.encodeURL(originalFileName));
    }
}
