package com.qczy.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FileToStringUtils {
    public static String readTextFile(String filePath) {
        BufferedReader reader;
        StringBuilder stringBuilder = new StringBuilder();
        
        try {
            reader = new BufferedReader(new FileReader(filePath));
            String line = reader.readLine();
            
            while (line != null) {
                stringBuilder.append(line);
                stringBuilder.append(System.lineSeparator());
                line = reader.readLine();
            }
            
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        return stringBuilder.toString();
    }
    
    public static void main(String[] args) {
        String filePath = "example.txt";
        String fileContent = readTextFile(filePath);
        System.out.println(fileContent);
    }
}