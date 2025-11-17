package com.qczy.utils;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.mapper.DataSonMapper;
import com.qczy.mapper.FileMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.entity.MarkInfoEntity;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/9/30 10:36
 * @Description: 文件下载工具类（JSON 格式写入）
 */
@Component
public class FileDownloadUtils {

    @Value("${upload.formalPath}")
    private String formalPath;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private FileMapper fileMapper;

    private final Gson gson = new GsonBuilder().setPrettyPrinting().create(); // 美化 JSON 格式

    public void writeFile(MarkInfoEntity markInfoEntity) {
        System.out.println("---------------------------------------------");
        System.out.println("进入方法中=");
        if (ObjectUtils.isEmpty(markInfoEntity)) {
            return;
        }

        String sonId = markInfoEntity.getSonId();
        if (StringUtils.isEmpty(sonId)) {
            return;
        }

        // 查询数据集信息
        DataSonEntity dataSonEntity = dataSonMapper.selectOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, sonId)
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return;
        }

        // 查询文件信息
        FileEntity fileEntity = fileMapper.selectById(markInfoEntity.getFileId());
        if (ObjectUtils.isEmpty(fileEntity)) {
            return;
        }

        String fileName = fileEntity.getFdName().split("\\.")[0]; // 提取文件名（不含扩展名）

        // 检查是否有需要写入的 JSON 数据
        if (StringUtils.isEmpty(markInfoEntity.getLabelMarkInfo())) {
            return;
        }

        // 构建文件目录
        String fileDir = formalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion() + "/json/";
        Path directory = Paths.get(fileDir);
        try {
            Files.createDirectories(directory); // 自动创建多级目录
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        // 写入 labelMarkInfo 到 .json 文件（美化 JSON 格式）
        writeJsonToFile(markInfoEntity.getLabelMarkInfo(), fileDir, fileName, ".json");
    }

    /**
     * 通用 JSON 写入方法（美化 JSON 格式，保留null值）
     */
    public void writeJsonToFile(String jsonContent, String dir, String baseName, String suffix) {
        if (StringUtils.isEmpty(jsonContent)) {
            return;
        }

        String filePath = dir + baseName + suffix;
        Path path = Paths.get(filePath);

        // 配置Gson：保留null值 + 美化格式
        Gson gson = new GsonBuilder()
                .serializeNulls() // 保留null值（关键配置）
                .setPrettyPrinting() // 美化JSON格式
                .create();

        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            // 解析 JSON 字符串并使用配置后的Gson美化格式
            JsonElement jsonElement = JsonParser.parseString(jsonContent);
            String prettyJson = gson.toJson(jsonElement);
            writer.write(prettyJson);
        } catch (Exception e) {
            // 如果解析失败，直接写入原始 JSON（保证数据完整）
            try (BufferedWriter fallbackWriter = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                fallbackWriter.write(jsonContent);
            } catch (IOException ex) {
                throw new RuntimeException("写入JSON文件失败：" + filePath, ex);
            }
        }
    }

}