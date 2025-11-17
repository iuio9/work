package com.qczy.controller.dataupload;


import com.qczy.common.result.Result;
import com.qczy.mapper.TempFileMapper;
import com.qczy.model.entity.TempFileEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/23 14:27
 * @Description: 大文件上传
 */
@RestController
@RequestMapping("/bigFileUpload/")
public class BigFileUploadController {




    @Value("${upload.tempPath}")
    private String tempPath;

    @Value("${upload.zipPath}")
    private String zipPath;

    @Autowired
    private TempFileMapper tempFileMapper;

    // 存储每个文件的分片计数器
    private ConcurrentHashMap<String, Integer> chunkCounters = new ConcurrentHashMap<>();
    // 存储每个文件的信号量
    private ConcurrentHashMap<String, Semaphore> semaphores = new ConcurrentHashMap<>();

    /**
     * 分片上传与自动合并接口
     */
    @PostMapping("/uploadChunk")
    public Result uploadChunk(
            @RequestParam("file") MultipartFile file,
            @RequestParam("index") int index,
            @RequestParam("fileHash") String fileHash,
            @RequestParam("chunkHash") String chuckHash,
            @RequestParam("chunkCount") int chunkCount,
            @RequestParam("fileName") String originalFilename // 接收原始文件名
    ) {
        try {
            // 创建分片存储目录
            String tempDir = zipPath + fileHash; // 分片存储目录按 fileHash 分类
            File dir = new File(tempDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            // 保存分片文件，确保路径指向具体文件
            File chunkFile = new File(tempDir, String.valueOf(index) + ".part"); // 为每个分片添加扩展名

            // 判断文件当前分片是否存在，如果存在，直接返回(断点续传！！！)
            if (chunkFile.exists()) {
                return Result.ok("Chunk " + index + " uploaded successfully");
            }


            System.out.println("Saving chunk to: " + chunkFile.getAbsolutePath());
            file.transferTo(chunkFile);

            Map<String, Object> data = new HashMap<>();

            // 更新分片计数器
            int uploadedChunks = incrementChunkCounter(fileHash);

            // 检查是否所有分片已上传完成
            if (uploadedChunks == chunkCount) {
                Semaphore semaphore = semaphores.computeIfAbsent(fileHash, k -> new Semaphore(1));
                try {
                    semaphore.acquire();
                    if (isAllChunksUploaded(new File(zipPath + fileHash), chunkCount)) {
                        Integer fileId = mergeChunks(fileHash, chunkCount, originalFilename);
                        data.put("status", 1);
                        data.put("id", fileId);
                        return Result.ok(data);
                    }
                } finally {
                    semaphore.release();
                }
            }

            data.put("status", 0);

            return Result.ok(data);
        } catch (Exception e) {
            e.printStackTrace();
            return Result.fail("Failed to upload chunk");
        }
    }

    /**
     * 增加分片计数器
     */
    private int incrementChunkCounter(String fileHash) {
        return chunkCounters.compute(fileHash, (k, v) -> v == null ? 1 : v + 1);
    }


    /**
     * 检查是否所有分片已上传完成
     */
    private boolean isAllChunksUploaded(File dir, int chunkCount) {
        String[] uploadedChunks = dir.list(); // 获取已上传的分片列表
        return uploadedChunks != null && uploadedChunks.length == chunkCount; // 判断分片数量是否与总数一致
    }


    // 修改合并方法
    private Integer mergeChunks(String fileHash, int chunkCount, String originalFilename) throws IOException {
        String tempDir = zipPath + fileHash;
        File mergedFile = new File(tempPath, originalFilename); // 使用原始文件名

        try (FileOutputStream fos = new FileOutputStream(mergedFile)) {
            for (int i = 0; i < chunkCount; i++) {
                File chunkFile = new File(tempDir, String.valueOf(i) + ".part");
                if (!chunkFile.exists()) {
                    throw new IOException("Missing chunk: " + i);
                }
                Files.copy(chunkFile.toPath(), fos);
            }
        }

        // 清理临时分片文件
        File dir = new File(tempDir);
        for (File file : dir.listFiles()) {
            file.delete();
        }
        dir.delete();

        // 记录临时文件夹存放地址
        TempFileEntity tempFileEntity = new TempFileEntity();
        tempFileEntity.setFdName(mergedFile.getName());
        tempFileEntity.setFdTempPath(tempPath + mergedFile.getName());
        tempFileEntity.setCreateTime(new Date());
        tempFileEntity.setFdSuffix(
                mergedFile.getName().substring(mergedFile.getName().lastIndexOf(".") + 1)
        );
        //  tempFileEntity.setFdType(Files.probeContentType(Paths.get(tempFileEntity.getFdTempPath())));
        // tempFileEntity.setFdSize();
        tempFileMapper.insert(tempFileEntity);
        System.out.println("File merged successfully: " + mergedFile.getAbsolutePath());
        return tempFileEntity.getId();


    }

}






