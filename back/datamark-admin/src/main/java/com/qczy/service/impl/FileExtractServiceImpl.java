package com.qczy.service.impl;

import cn.hutool.core.util.NumberUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.result.Result;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.service.FileExtractService;
import com.qczy.service.FileMarkService;
import com.qczy.utils.FileFormatSizeUtils;
import com.qczy.utils.ImageUtils;
import com.qczy.utils.MyHaoWebSocketUtils;
import net.coobird.thumbnailator.Thumbnails;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;

import java.io.*;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.regex.Pattern;

/**
 * 文件提取服务实现类
 * 处理压缩包解压、文件提取和特殊字符处理
 */
@Service
public class FileExtractServiceImpl implements FileExtractService {

    private static final Logger log = LoggerFactory.getLogger(FileExtractServiceImpl.class);

    // 用于匹配+和-的正则表达式
    private static final Pattern SPECIAL_CHAR_PATTERN = Pattern.compile("[+-]");

    @Autowired
    private TempFileMapper tempFileMapper;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private DataSonMapper dataSonMapper;
    @Autowired
    private MyHaoWebSocketUtils myHaoWebSocketUtils;
    @Autowired
    private DataImportLogMapper dataImportLogMapper;
    @Autowired
    private FileMarkService fileMarkService;
    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Value("${upload.formalPath}")
    private String formalPath;

    @Async
    @Override
    @Transactional
    public void fileExtract(DataSonEntityRequest dataSonEntityRequest) {
        if (ObjectUtils.isEmpty(dataSonEntityRequest) || StringUtils.isEmpty(dataSonEntityRequest.getFileIds())) {
            return;
        }

        // 开始时间
        Date startData = new Date();
        String[] fileIdsStr = dataSonEntityRequest.getFileIds().split(",");
        StringBuilder fileIds = new StringBuilder();
        // 文件总数量
        int sumCount = getZipSumCount(dataSonEntityRequest.getFileIds());  // 压缩包总数量
        log.info("zipCount = {}", sumCount);
        // 当前数量
        int currentCount = 0;
        // 结束时间
        Date endData = null;

        try {
            for (String fileIdStr : fileIdsStr) {
                int fileId = Integer.parseInt(fileIdStr);
                TempFileEntity tempFileEntity = tempFileMapper.selectById(fileId);
                if (ObjectUtils.isEmpty(tempFileEntity)) {
                    continue;
                }
                if (tempFileEntity.getFdName().endsWith(".zip")) {
                    Map<String, Object> map = unzip(tempFileEntity, dataSonEntityRequest, sumCount, currentCount);
                    fileIds.append(map.get("fileIds"));
                    currentCount = (int) map.get("currentCount");
                    sumCount = (int) map.get("sumCount");
                }
            }

            // 结束时间
            endData = new Date();

            // 判断是否有标注信息
            if (dataSonEntityRequest.getMarkStatus() == 1) {
                myHaoWebSocketUtils.sendMessage(dataSonEntityRequest.getSonId(), 50);
                fileMarkService.addMarkSon(dataSonEntityRequest, fileIds.deleteCharAt(fileIds.length() - 1).toString(), startData, sumCount, currentCount);
            }

            processData(dataSonEntityRequest, fileIds, startData, endData, sumCount);
        } catch (IOException e) {
            processData(dataSonEntityRequest, fileIds, startData, endData, sumCount);
            throw new RuntimeException(e);

        }
    }

    /**
     * 处理数据导入结果
     */
    public void processData(DataSonEntityRequest dataSonEntityRequest, StringBuilder fileIds, Date startData, Date endData, int sumCount) {
        try {
            // 首先先发送进度 //100%
            myHaoWebSocketUtils.sendMessage(dataSonEntityRequest.getSonId(), 100);

            if (dataSonEntityRequest.getMarkStatus() == 0) {
                // 修改数据集文件id
                DataSonEntity dataSonEntity = dataSonMapper.selectById(dataSonEntityRequest.getId());
                System.out.println(dataSonEntity);

                if (dataSonEntity != null) {
                    updateDataSonEntity(dataSonEntity, dataSonEntityRequest, fileIds, sumCount);
                    dataSonMapper.updateById(dataSonEntity);

                    // 记录数据导入日志
                    recordDataImportLog(dataSonEntityRequest, dataSonEntity, startData, endData);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    /**
     * 修改数据集信息
     */
    private void updateDataSonEntity(DataSonEntity dataSonEntity, DataSonEntityRequest dataSonEntityRequest, StringBuilder fileIds, int sumCount) {
        // 处理fileIds，确保不为空且最后一个字符可删除
        String processedFileIds = "";
        if (fileIds != null && fileIds.length() > 0) {
            // 只有当长度大于0时才删除最后一个字符（通常是多余的逗号）
            processedFileIds = fileIds.deleteCharAt(fileIds.length() - 1).toString();
        } else {
            // 日志记录空fileIds的情况，便于排查
            log.warn("fileIds is empty when updating DataSonEntity, sonId: {}", dataSonEntity.getSonId());
        }

        if (StringUtils.isEmpty(dataSonEntityRequest.getOldFileIds())) {  // 判断这个数据集之前有没有上传过图片
            dataSonEntity.setFileIds(processedFileIds);
            dataSonEntity.setStatus(0 + "% " + ("(" + 0 + "/" + sumCount + ")"));
        } else {
            // 处理旧文件ID和新文件ID拼接，避免空字符串导致的多余逗号
            String newFileIds;
            if (processedFileIds.isEmpty()) {
                newFileIds = dataSonEntityRequest.getOldFileIds();
            } else {
                newFileIds = dataSonEntityRequest.getOldFileIds() + "," + processedFileIds;
            }
            dataSonEntity.setFileIds(newFileIds);

            // 计算精度
            String[] files = newFileIds.split(",");
            int count = getMarkInfoCount(dataSonEntity.getSonId());
            int num = calculateAccuracy(count, files.length);
            dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + files.length + ")"));
        }
    }


    /**
     * 统计已经标注的数量
     */
    private int getMarkInfoCount(String sonId) {
        return markInfoMapper.selectCount(new LambdaQueryWrapper<MarkInfoEntity>().eq(MarkInfoEntity::getSonId, sonId));
    }

    /**
     * 计算标注完成百分比
     */
    private int calculateAccuracy(int count, int total) {
        return NumberUtil.div(Integer.toString(count), Integer.toString(total), 2)
                .multiply(BigDecimal.valueOf(100)).intValue();
    }

    /**
     * 记录数据导入日志
     */
    private void recordDataImportLog(DataSonEntityRequest dataSonEntityRequest, DataSonEntity dataSonEntity, Date startData, Date endData) {
        DataImportLogEntity importLogEntity = new DataImportLogEntity();
        importLogEntity.setFileSize(FileFormatSizeUtils.formatSize(0));
        importLogEntity.setStatus(1);  // 导入成功
        importLogEntity.setImportStartTime(startData);
        importLogEntity.setImportEndTime(endData);
        importLogEntity.setSonId(dataSonEntityRequest.getSonId());
        importLogEntity.setCreateTime(new Date());
        importLogEntity.setUserId(dataSonEntityRequest.getUserId());
        importLogEntity.setFileIds(dataSonEntity.getFileIds());
        dataImportLogMapper.insert(importLogEntity);
    }

    @Transactional
    public Map<String, Object> unzip(TempFileEntity tempFileEntity, DataSonEntityRequest dataSonEntityRequest,
                                     int sumCount, int currentCount) throws IOException {

        String filePath = tempFileEntity.getFdTempPath(); // 临时文件路径
        String destDir = formalPath + dataSonEntityRequest.getFatherId() + "/" + "v" + dataSonEntityRequest.getVersion() + "/" + "source" + "/"; // 解压目标路径

        File dir = new File(destDir);
        if (!dir.exists()) dir.mkdirs();

        // 记录正式文件的id
        StringBuilder fileIds = new StringBuilder();

        try (ZipFile zipFile = new ZipFile(new File(filePath), Charset.forName("GBK"))) {
            Enumeration<? extends ZipEntry> entries = zipFile.entries();

            while (entries.hasMoreElements()) {
                try {
                    ZipEntry entry = entries.nextElement();

                    // 获取纯文件名，不保留目录结构
                    String originalFileName = new File(entry.getName()).getName(); // 只获取文件名

                    // 处理特殊字符
                    String safeFileName = processSpecialChars(originalFileName);

                    File newFile = new File(destDir, safeFileName);

                    // 如果有重名文件，生成新的文件名
                    int counter = 1;
                    while (newFile.exists()) {
                        String baseName = safeFileName.replaceFirst("(\\.[^.]+$)", ""); // 去掉扩展名
                        String extension = safeFileName.replaceFirst(".*(\\.[^.]+$)", "$1");  // 恢复扩展名
                        String newFileName = baseName + "_" + counter++ + extension;
                        newFile = new File(destDir, newFileName);
                        safeFileName = newFileName;
                    }

                    // 如果是目录，跳过（不保留目录结构）
                    if (entry.isDirectory()) {
                        continue;
                    }

                    // 写入文件
                    try (InputStream inputStream = zipFile.getInputStream(entry);
                         FileOutputStream fos = new FileOutputStream(newFile)) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = inputStream.read(buffer)) > 0) {
                            fos.write(buffer, 0, len);
                        }
                    }

                    // 👉 关键：文件写入完成后，再判断是否为图片及是否损坏
                    // 此时newFile已存在且有内容，判断有效
                    if (ImageUtils.isImage(newFile.getPath())) {
                        // 判断图片是否损坏
                        if (ImageUtils.isImageCorrupted(newFile.getPath())) {
                            // 若损坏，删除无效文件，避免残留
                            if (newFile.exists()) {
                                boolean deleted = newFile.delete();
                                if (!deleted) {
                                    log.warn("损坏图片删除失败：{}", newFile.getAbsolutePath());
                                }
                            }
                            sumCount--; // 减少总计数
                            continue; // 跳过后续处理
                        }
                    }


                    // 判断文件类型
                    if (dataSonEntityRequest.getMarkStatus() == 0) {
                        // 不带标注信息上传，只能是图片
                        if (!ImageUtils.isImage(newFile.getPath())) {
                            // 非图片文件，删除并调整计数
                            if (newFile.exists()) {
                                newFile.delete();
                            }
                            sumCount--;
                            continue;
                        }
                    } else {
                        // 带标注信息上传，判断是否为允许的文件类型
                        if (!ImageUtils.isFile(newFile.getPath())) {
                            if (newFile.exists()) {
                                newFile.delete();
                            }
                            sumCount--;
                            continue;
                        }
                    }





                    // --------------------------------记录文件信息-----------------------------------
                    FileEntity fileEntity = new FileEntity();
                    fileEntity.setFdName(safeFileName);
                    fileEntity.setFdPath(destDir + safeFileName); // 使用安全文件名
                    fileEntity.setCreateTime(new Date());
                    fileEntity.setFileStatus(0);
                    fileEntity.setTaskId(null);

                    // ----------------------记录文件宽高 , 生成缩略图-------------------------
                    String shrinkPath = null;
                    // 生成缩略图
                    if (ImageUtils.isImage(fileEntity.getFdPath())) {
                        // 调用saveShrinkFile并传递3个参数
                        shrinkPath = saveShrinkFile(
                                formalPath + dataSonEntityRequest.getFatherId() + "/" + "v" + dataSonEntityRequest.getVersion(),
                                fileEntity.getFdPath(),
                                safeFileName
                        );

                        // 如果是图片，获取宽高
                        int[] dimensions = ImageUtils.getImageDimensions(fileEntity.getFdPath());
                        fileEntity.setWidth(dimensions[0]);
                        fileEntity.setHeight(dimensions[1]);
                        fileEntity.setHttpFilePath(shrinkPath);
                    }

                    // 新增数据到数据库
                    fileMapper.insert(fileEntity);
                    fileIds.append(fileEntity.getId()).append(",");

                    // 发送进度条， 使用webSocket 推送实时进度
                    currentCount++;

                    myHaoWebSocketUtils.sendMessage(
                            dataSonEntityRequest.getSonId(),
                            sumCount,  // 总数量
                            dataSonEntityRequest.getMarkStatus() == 0 ? currentCount : currentCount / 2); // 当前数量
                } catch (IllegalArgumentException e) {
                    System.err.println("Skipping malformed entry: " + e.getMessage());
                } catch (IOException e) {
                    System.err.println("Error processing entry: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Map<String, Object> result = new HashMap<>();
        result.put("fileIds", fileIds);
        result.put("currentCount", currentCount);
        result.put("sumCount", sumCount);
        return result;
    }

    // 文件缩放 - 修改为接收安全文件名
    @Transactional
    public String saveShrinkFile(
            String dirPath,
            String newFilePath,
            String safeFilename) {
        try {
            File file = new File(newFilePath);
            if (!file.exists()) {
                return null;
            }
            String newPathStr = dirPath + "/" + "shrin" + "/";
            File dirPathStr = new File(newPathStr);
            if (!dirPathStr.exists()) {
                dirPathStr.mkdirs();
            }

            // 开始新增缩放，使用安全文件名
            Thumbnails.of(file).
                    scale(0.1). // 图片缩放10%
                    outputQuality(1.0). // 图片质量100%
                    toFile(newPathStr + safeFilename);

            return newPathStr + safeFilename;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public int getZipSumCount(String tempFileIds) {
        int size = tempFileIds.split(",").length;  // 压缩包数量
        String[] zipFilePaths = new String[size];

        // 获取所有文件路径
        for (int i = 0; i < tempFileIds.split(",").length; i++) {
            TempFileEntity fileEntity = tempFileMapper.selectById(Integer.parseInt(tempFileIds.split(",")[i]));
            zipFilePaths[i] = fileEntity.getFdTempPath();
        }

        int totalFileCount = 0; // 总文件计数

        for (String filePath : zipFilePaths) {
            try {
                if (filePath.endsWith(".zip")) {
                    // 处理 ZIP 文件
                    totalFileCount += processZipFile(filePath);
                } else {
                    System.err.println("Unsupported file format: " + filePath);
                }
            } catch (Exception e) {
                System.err.println("Error processing file: " + filePath);
                e.printStackTrace();
            }
        }
        return totalFileCount;
    }

    // 处理 ZIP 文件
    public int processZipFile(String filePath) {
        int fileCount = 0;

        try (ZipFile zipFile = new ZipFile(filePath, Charset.forName("GBK"))) {
            Enumeration<? extends ZipEntry> entries = zipFile.entries();

            while (entries.hasMoreElements()) {
                try {
                    ZipEntry entry = entries.nextElement();
                    if (!entry.isDirectory()) {
                        fileCount++;
                        System.out.println("文件: " + entry.getName());
                    }
                } catch (IllegalArgumentException e) {
                    System.err.println("跳过格式错误的条目: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading ZIP file: " + filePath);
            e.printStackTrace();
        }

        return fileCount;
    }

    /**
     * 处理文件名中的+和-特殊字符
     */
    private String processSpecialChars(String filename) {
        if (filename == null) {
            return UUID.randomUUID().toString() + ".tmp";
        }

        // 使用正则表达式替换+和-为下划线
        String processedName = SPECIAL_CHAR_PATTERN.matcher(filename).replaceAll("_");

        // 确保扩展名正确保留
        String extension = getFileExtension(filename);
        if (!extension.isEmpty()) {
            processedName = processedName.substring(0, processedName.lastIndexOf('.')) + extension;
        }

        return processedName;
    }

    /**
     * 提取文件扩展名
     */
    private String getFileExtension(String filename) {
        if (filename == null) return "";
        int dotIndex = filename.lastIndexOf('.');
        return dotIndex > 0 ? filename.substring(dotIndex) : "";
    }
}