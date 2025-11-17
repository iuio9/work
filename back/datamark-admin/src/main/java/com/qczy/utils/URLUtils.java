package com.qczy.utils;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.net.URLDecoder;

public class URLUtils {

   /* public static String encodeURL(String url) {
        try {
            return URLEncoder.encode(url, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return url;
        }
    }

    public static String decodeURL(String url) {
        try {
            return URLDecoder.decode(url, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return url;
        }
    }

    public static void main(String[] args) {
        String originalURL = " with spaces/";
        String encodedURL = encodeURL(originalURL);
        System.out.println("Encoded URL: " + encodedURL);

        String decodedURL = decodeURL(encodedURL);
        System.out.println("Decoded URL: " + decodedURL);
    }*/






    public static String encodeURL(String baseURL){

        // 原始 URL 的基础部分
      //  String baseURL = "http://192.168.1.3:9092/formal//1288533794491465728/v1/source/35KV桑窝线016#塔-上相小号侧绝缘子涂层破损-DSC-51320.JPG";

        try {
            return URLEncoder.encode(baseURL, "UTF-8").replaceAll("\\+", "%20")  // 替换空格
                    .replace("%3A", ":")       // 恢复冒号 ":"
                    .replace("%2F", "/");

        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }
}