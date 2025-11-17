package com.qczy.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.ObjectUtils;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.service.FileThreadUploadService;
import com.qczy.utils.FileFormatSizeUtils;
import com.qczy.utils.ImageUtils;
import com.qczy.utils.MyHaoWebSocketUtils;
import com.qczy.utils.StringUtils;
import net.coobird.thumbnailator.Thumbnails;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.*;
import java.math.BigDecimal;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;

import static cn.hutool.core.util.NumberUtil.div;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/4/12 15:33
 * @Description: 修复特殊字符处理问题的文件上传服务
 */
@Service
public class FileThreadUploadServiceImpl implements FileThreadUploadService {

    private static final Logger logger = LoggerFactory.getLogger(FileServiceImpl.class);

    // 用于匹配+和-的正则表达式
    private static final Pattern SPECIAL_CHAR_PATTERN = Pattern.compile("[+-]");

    @Value("${upload.formalPath}")
    private String formalPath;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private TempFileMapper tempFileMapper;

    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Autowired
    private MyHaoWebSocketUtils myHaoWebSocketUtils;

    @Autowired
    private DataImportLogMapper dataImportLogMapper;

    private static final Object fileIdsLock = new Object();


    @Override
    @Async
    public void savaDataTempSonCopyFile(DataSonEntity dataSon, String sourceIdsStr) {
        if (ObjectUtils.isEmpty(dataSon)) {
            throw new RuntimeException("后端异常，（临时）数据集对象失败！");
        }
        if (StringUtils.isEmpty(formalPath)) {
            throw new RuntimeException("后端异常，文件上传路径为空！");
        }
        if (StringUtils.isEmpty(dataSon.getFileIds())) {
            return;
        }

        // 记录数据导入日志
        DataImportLogEntity importLogEntity = new DataImportLogEntity();

        // 拼接正式上传文件的路径
        String newFilePath = formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion() + "/" + "source" + "/";
        File fileMkdir = new File(newFilePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            if (!fileMkdir.mkdirs()) {
                throw new RuntimeException("后端异常，创建文件目录失败！");
            }
        }

        String[] tempFiles = dataSon.getFileIds().split(",");
        // 记录总的文件大小
        AtomicLong size = new AtomicLong();
        // 记录正式文件的id
        StringBuilder fileIds = new StringBuilder();
        AtomicInteger currentIndex = new AtomicInteger(0);
        Date startDate = new Date();
        // 创建线程池
        int corePoolSize = Runtime.getRuntime().availableProcessors();
        int maxPoolSize = corePoolSize * 2;
        int queueCapacity = 100;
        ExecutorService executorService = new ThreadPoolExecutor(
                corePoolSize,
                maxPoolSize,
                0L,
                TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(queueCapacity),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        List<Future<Integer>> futures = new ArrayList<>();

        for (String tempFileId : tempFiles) {
            final String safeTempFileId = tempFileId; // 确保lambda表达式中变量的有效性
            futures.add(executorService.submit(() -> {
                try {
                    // 开始进行上传
                    TempFileEntity tempFileEntity = tempFileMapper.selectById(Integer.parseInt(safeTempFileId));
                    if (ObjectUtils.isEmpty(tempFileEntity)) {
                        throw new RuntimeException("后端异常，临时文件不存在！");
                    }

                    // 临时文件的路径
                    File tempFile = new File(tempFileEntity.getFdTempPath());
                    long length = tempFile.length();
                    size.addAndGet(length);

                    // 处理特殊字符，生成安全文件名
                    String safeFilename = processSpecialChars(tempFileEntity.getFdName());

                    // 使用安全文件名进行文件复制
                    optimizedFileCopy(tempFile.getPath(), newFilePath + "/" + safeFilename);

                    String shrinkPath = null;
                    // 生成缩略图
                    if (ImageUtils.isImage(tempFileEntity.getFdTempPath())) {
                        shrinkPath = saveShrinkFile(
                                formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion(),
                                newFilePath + "/" + safeFilename, // 使用安全文件名
                                safeFilename // 传递安全文件名
                        );
                    }

                    // 同步文件数据库
                    int fileId = savaFile(tempFileEntity, dataSon.getFatherId(), dataSon.getVersion(), shrinkPath, safeFilename);
                    synchronized (fileIdsLock) {
                        if (fileId > 0) {
                            fileIds.append(fileId).append(",");
                        }
                    }

                    // 使用websocket 发送信息
                    myHaoWebSocketUtils.sendMessage(dataSon.getSonId(), tempFiles.length, currentIndex.incrementAndGet());

                    return fileId;
                } catch (IOException e) {
                    logger.error("文件操作异常", e);
                    importLogEntity.setStatus(2);  // 导入失败
                    try {
                        dataImportLogMapper.updateById(importLogEntity);
                    } catch (Exception ex) {
                        logger.error("更新导入日志失败", ex);
                    }
                    dataSon.setStatus("0% (0/0)");
                    try {
                        dataSonMapper.updateById(dataSon);
                    } catch (Exception ex) {
                        logger.error("更新数据集状态失败", ex);
                    }
                    throw new RuntimeException(e);
                }
            }));
        }

        // 等待所有任务完成
        for (Future<Integer> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                logger.error("线程被中断", e);
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                logger.error("任务执行异常", e);
                throw new RuntimeException(e);
            }
        }

        // 关闭线程池
        executorService.shutdown();
        Date endDate = new Date();
        String ids;
        synchronized (fileIdsLock) {
            if (fileIds.length() > 0) {
                ids = fileIds.deleteCharAt(fileIds.length() - 1).toString();
            } else {
                ids = "";
            }
        }
        processFileImport(dataSon, sourceIdsStr, ids, size.get(), startDate, endDate, importLogEntity);

        logger.info("接口总耗时：" + (endDate.getTime() - startDate.getTime()) + "ms");
    }

    // 处理特殊字符，将+和-替换为下划线
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

    // 提取文件扩展名
    private String getFileExtension(String filename) {
        if (filename == null) return "";
        int dotIndex = filename.lastIndexOf('.');
        return dotIndex > 0 ? filename.substring(dotIndex) : "";
    }

    // 文件拷贝
    public static void optimizedFileCopy(String sourceFilePath, String targetFilePath) throws IOException {
        try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(sourceFilePath));
             BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(targetFilePath))) {
            byte[] buffer = new byte[8192]; // 使用 8KB 的缓冲区
            int readLen;
            while ((readLen = in.read(buffer)) != -1) {
                out.write(buffer, 0, readLen);
            }
        }
    }

    // 文件缩放 - 修改为接收安全文件名
    public static String saveShrinkFile(
            String dirPath,
            String newFilePath,
            String safeFilename) {
        try {
            File file = new File(newFilePath);
            if (!file.exists()) { // 判断文件是否存在
                return null;
            }
            String newPathStr = dirPath + "/" + "shrin" + "/";
            File dirPathStr = new File(newPathStr);
            if (!dirPathStr.exists()) { //判断当前目录是否存在
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

    // 修改savaFile方法，接收安全文件名
    public int savaFile(TempFileEntity tempFileEntity, String groupId, Integer version, String shrinkPath, String safeFilename) {
        FileEntity fileEntity = new FileEntity();
        try {
            BeanUtils.copyProperties(tempFileEntity, fileEntity);

            // 使用安全文件名构建文件路径
            fileEntity.setFdPath(formalPath + groupId + "/" + "v" + version + "/" + "source" + "/" + safeFilename);
            fileEntity.setCreateTime(new Date());
            fileEntity.setFileStatus(0);
            fileEntity.setTaskId("");

            //----------------------记录文件宽高-------------------------
            if (!fileEntity.getFdSuffix().equals(".json")) {
                fileEntity.setHttpFilePath(shrinkPath);

                // 检查文件是否存在
                File imageFile = new File(fileEntity.getFdPath());
                if (imageFile.exists()) {
                    Image image = ImageIO.read(imageFile);
                    fileEntity.setWidth(image.getWidth(null));
                    fileEntity.setHeight(image.getHeight(null));
                } else {
                    logger.warn("图片文件不存在，无法获取宽高: {}", fileEntity.getFdPath());
                }
            }
            //---------------------------------------------------------

            int result = fileMapper.insert(fileEntity);
            if (result < SystemConstant.MAX_SIZE) {
                throw new RuntimeException("后端异常，文件数据库同步失败！");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return fileEntity.getId();
    }

    // 未使用的方法，可以考虑删除
    public int savaFile1(FileEntity tempFileEntity, String groupId, Integer version, String shrinkPath) {
        // 此方法未被使用，可以考虑删除
        return 0;
    }

    public void processFileImport(DataSonEntity dataSon, String sourceIdsStr, String ids, long size, Date startDate, Date endDate, DataImportLogEntity importLogEntity) {
        // 记录文件信息
        if (StringUtils.isEmpty(sourceIdsStr)) {
            String[] fileSizeArray = dataSon.getFileIds() != null ? dataSon.getFileIds().split(",") : new String[0];
            dataSon.setFileIds(ids);
            dataSon.setStatus("0% (0/" + fileSizeArray.length + ")");
        } else {
            dataSon.setFileIds(sourceIdsStr + "," + ids);
            if (!ObjectUtils.isEmpty(dataSon)) {
                String[] files = dataSon.getFileIds() != null ? dataSon.getFileIds().split(",") : new String[0];
                Integer count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, dataSon.getSonId())
                );
                int num = div(count != null ? count.toString() : "0", Integer.toString(files.length), 2)
                        .multiply(BigDecimal.valueOf(100)).intValue();
                dataSon.setStatus(num + "% " + ("(" + count + "/" + files.length + ")"));
            }
        }

        // 更新数据库记录
        if (dataSon != null) {
            dataSonMapper.updateById(dataSon);
        }

        // 记录导入日志
        importLogEntity.setFileSize(FileFormatSizeUtils.formatSize(size));
        importLogEntity.setStatus(1);  // 导入成功
        importLogEntity.setImportStartTime(startDate);
        importLogEntity.setImportEndTime(endDate);
        importLogEntity.setSonId(dataSon.getSonId());
        importLogEntity.setCreateTime(new Date());
        importLogEntity.setUserId(dataSon.getUserId());
        importLogEntity.setFileIds(ids);
        // 记录数据导入数据库
        dataImportLogMapper.insert(importLogEntity);
    }
}