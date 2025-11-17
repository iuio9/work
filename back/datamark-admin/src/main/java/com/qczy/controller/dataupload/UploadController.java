package com.qczy.controller.dataupload;

import com.qczy.common.result.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.io.*;
import java.util.UUID;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/21 9:53
 * @Description:
 */
@RestController
public class UploadController {

    private static final Logger log = LoggerFactory.getLogger(UploadController.class);

    @Value("${upload.modelDescPath}")
    private String modelDescPath;

    @Value("${upload.modelCasePath}")
    private String modelCasePath;

    @Value("${upload.pdfPath}")
    private String pdfPath;

    @PostMapping("/upload")
    public Result upload(@RequestParam("file") MultipartFile file, HttpServletRequest request) {
        int type = Integer.parseInt(request.getHeader("type"));
        String dirPath = getDirPathByType(type);
        if (dirPath == null) {
            return Result.fail("请选择要上传的文件类型！");
        }

        // 确保目录存在
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // 生成唯一文件名
        String originalFilename = file.getOriginalFilename();
        String uniqueFilename = generateUniqueFilename(dirPath, originalFilename);

        try (InputStream in = file.getInputStream();
             FileOutputStream out = new FileOutputStream(new File(dirPath, uniqueFilename))) {

            byte[] buf = new byte[8192];
            int readLen;
            while ((readLen = in.read(buf)) != -1) {
                out.write(buf, 0, readLen);
            }

        } catch (Exception e) {
            log.error("文件上传失败", e);
            return Result.fail("上传失败：" + e.getMessage());
        }

        log.info("文件上传成功：{}", uniqueFilename);
        return Result.ok(dirPath + uniqueFilename);
    }

    /**
     * 根据文件类型获取存储路径
     */
    private String getDirPathByType(int type) {
        switch (type) {
            case 1: return modelDescPath;
            case 2: return modelCasePath;
            case 3: return pdfPath;
            default: return null;
        }
    }

    /**
     * 生成唯一文件名（原文件名_UUID.扩展名）
     */
    private String generateUniqueFilename(String dirPath, String originalFilename) {
        if (originalFilename == null || originalFilename.isEmpty()) {
            return UUID.randomUUID().toString();
        }

        // 分离文件名和扩展名
        int dotIndex = originalFilename.lastIndexOf('.');
        String name = dotIndex > 0 ? originalFilename.substring(0, dotIndex) : originalFilename;
        String ext = dotIndex > 0 ? originalFilename.substring(dotIndex) : "";

        // 生成带UUID的唯一文件名
        String uniqueName = name + "_" + UUID.randomUUID().toString() + ext;

        return uniqueName;
    }
}