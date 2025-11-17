package com.qczy.service.impl;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.NumberUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.idrsolutions.image.png.PngCompressor;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.result.Result;
import com.qczy.mapper.*;
import com.qczy.model.entity.*;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.ResultDataSonRequest;
import com.qczy.service.DataSonService;
import com.qczy.service.FileService;
import com.qczy.utils.*;
import net.coobird.thumbnailator.Thumbnails;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.ObjectUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.math.BigDecimal;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Objects;
import java.util.UUID;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 文件服务实现类，包含特殊字符处理功能
 */
@Service
@SuppressWarnings({"all"})  // 抑制所有警告
public class FileServiceImpl extends ServiceImpl<FileMapper, FileEntity> implements FileService {

    private static final Logger log = LoggerFactory.getLogger(FileServiceImpl.class);
    // 用于匹配+和-的正则表达式
    private static final Pattern SPECIAL_CHAR_PATTERN = Pattern.compile("[+-]");

    @Value("${file.accessAddress}")
    private String accessAddress;

    @Autowired
    private FileMapper fileMapper;

    @Autowired
    private TempFileMapper tempFileMapper;

    @Autowired
    private DataImportLogMapper dataImportLogMapper;

    @Value("${upload.tempPath}")
    private String tempPath;

    @Value("${upload.formalPath}")
    private String formalPath;
    @Autowired
    private DataSonMapper dataSonMapper;

    @Autowired
    private MarkInfoMapper markInfoMapper;

    @Autowired
    private MyHaoWebSocketUtils myHaoWebSocketUtils;

    @Autowired
    private FileAndJsonUtils fileAndJsonUtils;

    @Async
    @Transactional
    @Override
    public void savaDataTempSonCopyFile(DataSonEntity dataSon, String sourceIdsStr) throws IOException {
        if (ObjectUtils.isEmpty(dataSon)) {
            throw new RuntimeException("后端异常，（临时）数据集对象失败！");
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
            fileMkdir.mkdirs();
        }

        FileInputStream in = null;
        FileOutputStream out = null;

        String[] tempFiles = dataSon.getFileIds().split(",");

        Date startDate = new Date();
        log.info("--------------------------------导入开始时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
        // 记录总的文件大小
        long size = 0;
        // 记录正式文件的id
        StringBuilder fileIds = new StringBuilder();
        for (int i = 0; i < tempFiles.length; i++) {
            // 开始进行上传
            TempFileEntity tempFileEntity = tempFileMapper.selectById(Integer.parseInt(tempFiles[i]));
            log.info("-------------------------------" + Integer.parseInt(tempFiles[i]));
            if (ObjectUtils.isEmpty(tempFileEntity)) {
                throw new RuntimeException("后端异常，临时文件不存在！");
            }

            // 处理特殊字符
            String originalFilename = tempFileEntity.getFdName();
            String safeFilename = processSpecialChars(originalFilename);

            // 临时文件的路径
            try {
                in = new FileInputStream(tempFileEntity.getFdTempPath());

                long length = new File(tempFileEntity.getFdTempPath()).length();
                log.info("--------------------------------当前文件大小>" + length);
                size += length;

                log.info("--------------------------------in>" + tempFileEntity.getFdTempPath());
                out = new FileOutputStream(newFilePath + "/" + safeFilename); // 使用安全文件名
                log.info("--------------------------------out>" + newFilePath + "/" + safeFilename);
                // 一次读1024个字节
                byte[] buf = new byte[1024];
                int readLen = 0;

                while ((readLen = in.read(buf)) != -1) {
                    // 读到数据以后，就写入
                    out.write(buf, 0, readLen);
                }

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
                log.info("--------------------------------文件id为>" + fileId);
                fileIds.append(fileId).append(",");

                // 使用websocket 发送信息
                myHaoWebSocketUtils.sendMessage(dataSon.getSonId(), (tempFiles.length), (i + 1));

            } catch (IOException e) {
                importLogEntity.setStatus(2);  // 导入失败
                dataImportLogMapper.updateById(importLogEntity);
                dataSon.setStatus("0% (0/0)");
                dataSonMapper.updateById(dataSon);

                throw new RuntimeException(e);
            } finally {
                if (in != null) {
                    in.close();
                }
                if (out != null) {
                    out.close();
                }
            }
        }

        String ids = fileIds.deleteCharAt(fileIds.length() - 1).toString();

        // 记录文件信息
        if (StringUtils.isEmpty(sourceIdsStr)) {
            String[] fileSize = dataSon.getFileIds().split(",");
            dataSon.setFileIds(ids);
            dataSon.setStatus("0% (0/" + fileSize.length + ")");
        } else {
            dataSon.setFileIds(sourceIdsStr + "," + ids);
            if (!ObjectUtils.isEmpty(dataSon)) {
                String[] files = dataSon.getFileIds().split(",");
                Integer count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, dataSon.getSonId())
                );
                int num = NumberUtil.div(count.toString(), Integer.toString(files.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
                dataSon.setStatus(num + "% " + ("(" + count + "/" + files.length + ")"));
            }

        }


        dataSonMapper.updateById(dataSon);


        Date endDate = new Date();
        log.info("--------------------------------导入结束时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
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


        log.info("-----------------Upload File Successfully-----------------");
    }

    // 文件缩放
    @Transactional
    private String saveShrinkFile(
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

    @Async
    @Transactional
    @Override
    public void savaDataSonCopyFile(DataSonEntity dataSon) throws IOException {
        if (ObjectUtils.isEmpty(dataSon)) {
            throw new RuntimeException("后端异常，（临时）数据集对象失败！");
        }

        if (StringUtils.isEmpty(dataSon.getFileIds())) {
            throw new RuntimeException("后端异常，传入的文件id为空！");
        }

        // 记录数据导入日志
        DataImportLogEntity importLogEntity = new DataImportLogEntity();


        // 拼接正式上传文件的路径
        String newFilePath = formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion() + "/" + "source" + "/";
        File fileMkdir = new File(newFilePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }

        FileInputStream in = null;
        FileOutputStream out = null;

        String[] files = dataSon.getFileIds().split(",");

        Date startDate = new Date();
        log.info("--------------------------------导入开始时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
        // 记录总的文件大小
        long size = 0;
        // 记录正式文件的id
        StringBuilder fileIds = new StringBuilder();
        for (int i = 0; i < files.length; i++) {
            // 开始进行上传
            // 判断是从正式表里面拷贝，还是从临时表里面拷贝
            FileEntity fileEntity = fileMapper.selectById(Integer.parseInt(files[i]));
            log.info("-------------------------------" + Integer.parseInt(files[i]));
            if (ObjectUtils.isEmpty(fileEntity)) {
                throw new RuntimeException("后端异常，临时文件不存在！");
            }

            // 处理特殊字符
            String originalFilename = fileEntity.getFdName();
            String safeFilename = processSpecialChars(originalFilename);

            // 临时文件的路径
            try {
                in = new FileInputStream(fileEntity.getFdPath());

                long length = new File(fileEntity.getFdPath()).length();
                log.info("--------------------------------当前文件大小>" + length);
                size += length;

                log.info("--------------------------------in>" + fileEntity.getFdPath());
                out = new FileOutputStream(newFilePath + "/" + safeFilename); // 使用安全文件名
                log.info("--------------------------------out>" + newFilePath + "/" + safeFilename);
                // 一次读1024个字节
                byte[] buf = new byte[1024];
                int readLen = 0;

                while ((readLen = in.read(buf)) != -1) {
                    // 读到数据以后，就写入
                    out.write(buf, 0, readLen);
                }

                String shrinkPath = null;
                // 生成缩略图
                if (ImageUtils.isImage(fileEntity.getFdPath())) {
                    shrinkPath = saveShrinkFile(
                            formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion(),
                            newFilePath + "/" + safeFilename, // 使用安全文件名
                            safeFilename // 传递安全文件名
                    );
                }

                // 同步文件数据库
                int fileId = savaFile(fileEntity, dataSon.getFatherId(), dataSon.getVersion(), shrinkPath, safeFilename);
                log.info("--------------------------------文件id为>" + fileId);
                fileIds.append(fileId).append(",");

                // 使用websocket 发送信息
                myHaoWebSocketUtils.sendMessage(dataSon.getSonId(), (files.length), (i + 1));

            } catch (IOException e) {
                importLogEntity.setStatus(2);  // 导入失败
                throw new RuntimeException(e);
            } finally {
                if (in != null) {
                    in.close();
                }
                if (out != null) {
                    out.close();
                }
            }
        }

        String ids = fileIds.deleteCharAt(fileIds.length() - 1).toString();


        // 记录文件信息
        dataSon.setFileIds(ids);
        if (!ObjectUtils.isEmpty(dataSon)) {
            String[] files1 = dataSon.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSon.getSonId())
            );
            int num = NumberUtil.div(count.toString(), Integer.toString(files1.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
            dataSon.setStatus(num + "% " + ("(" + count + "/" + files1.length + ")"));
        }
        dataSonMapper.updateById(dataSon);


        Date endDate = new Date();
        log.info("--------------------------------导入结束时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
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


        log.info("-----------------Upload File Successfully-----------------");
    }

    @Async
    @Override
    @Transactional
    public void savaDataSonCopyFile1(DataSonEntityRequest dataSon, ResultDataSonRequest request) throws IOException {
        if (ObjectUtils.isEmpty(dataSon)) {
            throw new RuntimeException("后端异常，（临时）数据集对象失败！");
        }

        if (StringUtils.isEmpty(dataSon.getFileIds())) {
            throw new RuntimeException("后端异常，传入的文件id为空！");
        }

        // 记录数据导入日志
        DataImportLogEntity importLogEntity = new DataImportLogEntity();


        // 拼接正式上传文件的路径
        String newFilePath = formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion() + "/" + "source" + "/";
        File fileMkdir = new File(newFilePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }

        FileInputStream in = null;
        FileOutputStream out = null;

        String[] files = dataSon.getFileIds().split(",");

        Date startDate = new Date();
        log.info("--------------------------------导入开始时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
        // 记录总的文件大小
        long size = 0;
        // 记录正式文件的id
        StringBuilder fileIds = new StringBuilder();
        for (int i = 0; i < files.length; i++) {
            // 开始进行上传
            // 判断是从正式表里面拷贝，还是从临时表里面拷贝
            FileEntity fileEntity = fileMapper.selectById(Integer.parseInt(files[i]));
            log.info("-------------------------------" + Integer.parseInt(files[i]));
            if (ObjectUtils.isEmpty(fileEntity)) {
                throw new RuntimeException("后端异常，临时文件不存在！");
            }

            // 处理特殊字符
            String originalFilename = fileEntity.getFdName();
            String safeFilename = processSpecialChars(originalFilename);

            // 临时文件的路径
            try {
                in = new FileInputStream(fileEntity.getFdPath());

                long length = new File(fileEntity.getFdPath()).length();
                log.info("--------------------------------当前文件大小>" + length);
                size += length;

                log.info("--------------------------------in>" + fileEntity.getFdPath());
                out = new FileOutputStream(newFilePath + "/" + safeFilename); // 使用安全文件名
                log.info("--------------------------------out>" + newFilePath + "/" + safeFilename);
                // 一次读1024个字节
                byte[] buf = new byte[1024];
                int readLen = 0;

                while ((readLen = in.read(buf)) != -1) {
                    // 读到数据以后，就写入
                    out.write(buf, 0, readLen);
                }

                String shrinkPath = null;
                // 生成缩略图
                if (ImageUtils.isImage(fileEntity.getFdPath())) {
                    shrinkPath = saveShrinkFile(
                            formalPath + dataSon.getFatherId() + "/" + "v" + dataSon.getVersion(),
                            newFilePath + "/" + safeFilename, // 使用安全文件名
                            safeFilename // 传递安全文件名
                    );
                }

                // 同步文件数据库
                int fileId = savaFile1(fileEntity, dataSon.getFatherId(), dataSon.getVersion(), shrinkPath, safeFilename);
                log.info("--------------------------------文件id为>" + fileId);
                fileIds.append(fileId).append(",");

                // 使用websocket 发送信息
                myHaoWebSocketUtils.sendMessage(dataSon.getSonId(), (files.length), (i + 1));

            } catch (IOException e) {
                importLogEntity.setStatus(2);  // 导入失败
                throw new RuntimeException(e);
            } finally {
                if (in != null) {
                    in.close();
                }
                if (out != null) {
                    out.close();
                }
            }
        }

        String ids = fileIds.deleteCharAt(fileIds.length() - 1).toString();
        dataSon.setFileIds(ids);

        // 记录文件信息
        if (!ObjectUtils.isEmpty(dataSon)) {
            String[] files1 = dataSon.getFileIds().split(",");
            Integer count = markInfoMapper.selectCount(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSon.getSonId())
            );
            int num = NumberUtil.div(count.toString(), Integer.toString(files1.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
            dataSon.setStatus(num + "% " + ("(" + count + "/" + files1.length + ")"));
        }
        dataSonMapper.updateById(dataSon);


        Date endDate = new Date();
        log.info("--------------------------------导入结束时间>" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(startDate));
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


        log.info("-----------------Upload File Successfully-----------------");


        // 判断任务类型是否为缺陷生成，如果是缺陷生成，则进行图片与json合并
        fileAndJsonUtils.setFlawFileAndJson(request, dataSon);
    }

    @Transactional
    public int savaFile(TempFileEntity tempFileEntity, String groupId, Integer version, String shrinkPath, String safeFilename) {
        FileEntity fileEntity = new FileEntity();
        try {
            BeanUtils.copyProperties(tempFileEntity, fileEntity);
            fileEntity.setFdPath(formalPath + groupId + "/" + "v" + version + "/" + "source" + "/" + safeFilename); // 使用安全文件名
            fileEntity.setCreateTime(new Date());
            fileEntity.setFileStatus(0);
            //----------------------记录文件宽高-------------------------
            if (!fileEntity.getFdSuffix().equals(".json")) {
                fileEntity.setHttpFilePath(shrinkPath);
                Image image = ImageIO.read(new File(fileEntity.getFdPath()));
                fileEntity.setWidth(image.getWidth(null));
                fileEntity.setHeight(image.getHeight(null));
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

    @Transactional
    public int savaFile(FileEntity fileEntity, String groupId, Integer version, String shrinkPath, String safeFilename) {
        FileEntity newFileEntity = new FileEntity();
        try {
            BeanUtils.copyProperties(fileEntity, newFileEntity);
            newFileEntity.setFdPath(formalPath + groupId + "/" + "v" + version + "/" + "source" + "/" + safeFilename); // 使用安全文件名
            newFileEntity.setCreateTime(new Date());
            newFileEntity.setFileStatus(0);
            //----------------------记录文件宽高-------------------------
            // 判断是否上传的是图片，如果为图片，则获取宽高
            if (ImageUtils.isImage(newFileEntity.getFdPath())) {
                int[] ints = ImageUtils.getImageDimensions(newFileEntity.getFdPath());
                newFileEntity.setWidth(ints[0]);
                newFileEntity.setHeight(ints[1]);
                newFileEntity.setHttpFilePath(shrinkPath);
            }
            //---------------------------------------------------------


            int result = fileMapper.insert(newFileEntity);

            if (result < SystemConstant.MAX_SIZE) {
                throw new RuntimeException("后端异常，文件数据库同步失败！");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return newFileEntity.getId();
    }

    @Transactional
    public int savaFile1(FileEntity fileEntity, String groupId, Integer version, String shrinkPath, String safeFilename) {
        FileEntity newFileEntity = new FileEntity();
        try {
            BeanUtils.copyProperties(fileEntity, newFileEntity);
            newFileEntity.setFdPath(formalPath + groupId + "/" + "v" + version + "/" + "source" + "/" + safeFilename); // 使用安全文件名
            newFileEntity.setCreateTime(new Date());
            newFileEntity.setFileStatus(0);
            newFileEntity.setTaskId(null);
            //----------------------记录文件宽高-------------------------
            // 判断是否上传的是图片，如果为图片，则获取宽高
            if (ImageUtils.isImage(newFileEntity.getFdPath())) {
                int[] ints = ImageUtils.getImageDimensions(newFileEntity.getFdPath());
                newFileEntity.setWidth(ints[0]);
                newFileEntity.setHeight(ints[1]);
                newFileEntity.setHttpFilePath(shrinkPath);
            }

            //---------------------------------------------------------

            int result = fileMapper.insert(newFileEntity);

            if (result < SystemConstant.MAX_SIZE) {
                throw new RuntimeException("后端异常，文件数据库同步失败！");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return newFileEntity.getId();
    }

    @Override
    public int deleteFile(String groupId) throws IOException {
        return deleteFileCascade(formalPath + groupId) == true ? 1 : 0;
    }

    @Override
    public int deleteFile(String groupId, String version, String scource) throws IOException {
        return deleteFileCascade(formalPath + groupId + "/" + "v" + version + "/scource") == true ? 1 : 0;
    }

    @Override
    public int deleteFile(String groupId, String version) throws IOException {
        return deleteFileCascade(formalPath + groupId + "/" + "v" + version) == true ? 1 : 0;
    }

    @Override
    public String addAlgorithmFile(File file, String dir) {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {
            FileEntity fileEntity = new FileEntity();

            // 处理特殊字符
            String originalFilename = file.getName();
            String safeFilename = processSpecialChars(originalFilename);

            fileEntity.setFdPath(dir + "/" + safeFilename);
            fileEntity.setFdName(safeFilename);
            fileEntity.setFdType(Files.probeContentType(file.toPath()));
            fileEntity.setFdSize(FileFormatSizeUtils.formatSize(file.length()));
            fileEntity.setFdSuffix(Objects.requireNonNull(safeFilename).substring(safeFilename.lastIndexOf(".")));
            fileEntity.setFdAccessPath(safeFilename);
            if (!fileEntity.getFdSuffix().equals(".json")) {
                //----------------------记录文件宽高-------------------------
                Image image = ImageIO.read(new File(fileEntity.getFdPath()));
                fileEntity.setWidth(image.getWidth(null));
                fileEntity.setHeight(image.getHeight(null));
                //---------------------------------------------------------
            }

            int save = fileMapper.insert(fileEntity);
            if (save > 0) {
                String fileDirOrName = fileEntity.getFdPath();
                String prefixToRemove = formalPath;
                // 使用 replaceFirst 方法去掉前缀
                fileDirOrName = accessAddress + fileDirOrName.substring(prefixToRemove.length());
                return fileDirOrName;
            }


        } catch (IOException e) {
            e.printStackTrace();
            return null;
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        return null;
    }

    @Override
    public String addAlgorithmFile(File file) {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {

            if (!FileUtil.isDirectory(file)) {
                FileUtil.mkdir(file);
            }
            String originalFilename = file.getName();
            String safeFilename = processSpecialChars(originalFilename);
            String file1 = formalPath + "/" + safeFilename;
//            if (FileUtil.exist(file1)) {
//                String fileDirOrName = fileEntity.getFdPath();
//                String prefixToRemove = formalPath;
//
//                // 使用 replaceFirst 方法去掉前缀
//                fileDirOrName = accessAddress+ fileDirOrName.replaceFirst("^" + prefixToRemove, "");
//                return fileDirOrName+fileEntity.getFdName();
//            }
            inputStream = new FileInputStream(file);

            outputStream = new FileOutputStream(file1);
            int read;
            byte[] bytes = new byte[1024];
            while ((read = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, read);
            }
            ;

            FileEntity fileEntity = new FileEntity();

            fileEntity.setFdPath(file1);
            fileEntity.setFdName(safeFilename);
            fileEntity.setFdType(Files.probeContentType(file.toPath()));
            fileEntity.setFdSize(FileFormatSizeUtils.formatSize(file.length()));
            fileEntity.setFdSuffix(Objects.requireNonNull(safeFilename).substring(safeFilename.lastIndexOf(".")));
            fileEntity.setFdAccessPath(safeFilename);

            //----------------------记录文件宽高-------------------------
            Image image = ImageIO.read(new File(fileEntity.getFdPath()));
            fileEntity.setWidth(image.getWidth(null));
            fileEntity.setHeight(image.getHeight(null));
            //---------------------------------------------------------

            int save = fileMapper.insert(fileEntity);
            if (save > 0) {
                String fileDirOrName = fileEntity.getFdPath();
                String prefixToRemove = formalPath;

                // 使用 replaceFirst 方法去掉前缀
                fileDirOrName = accessAddress + fileDirOrName.replaceFirst("^" + prefixToRemove, "") + "/";
                return fileDirOrName + fileEntity.getFdName();
            }


        } catch (IOException e) {
            e.printStackTrace();
            return null;
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        return null;
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

    /**
     * 删除目录（文件夹）以及目录下的文件
     *
     * @param sPath 被删除目录的文件路径
     * @return 目录删除成功返回true，否则返回false
     */
    public static boolean deleteFileCascade(String fileStr) throws IOException {
        log.info("--------------->" + fileStr);
        File file = new File(fileStr);
        if (!file.exists()) {
            return false;
        }

        if (file.isFile()) {
            return file.delete();
        }

        Files.walkFileTree(Paths.get(fileStr), new SimpleFileVisitor<Path>() {
            /**
             * 遍历到某个文件时执行
             */
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file); //删除文件
                return super.visitFile(file, attrs);
            }

            /**
             * 当退出某个目录时执行此方法
             */
            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);//删除目录
                return super.postVisitDirectory(dir, exc);
            }
        });

        return true;
    }
}