package com.qczy.controller.dataupload;

import cn.hutool.core.util.NumberUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.constant.SystemConstant;
import com.qczy.common.result.Result;
import com.qczy.mapper.DataSonMapper;
import com.qczy.mapper.FileMapper;
import com.qczy.mapper.MarkInfoMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.entity.MarkInfoEntity;
import com.qczy.model.request.FileDeleteRequest;
import com.qczy.model.request.MarkFileRequest;
import com.qczy.service.DataSonService;
import com.qczy.service.FileService;
import com.qczy.utils.FileFormatSizeUtils;
import io.swagger.annotations.Api;
import net.coobird.thumbnailator.Thumbnails;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.util.ObjectUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.util.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/23 9:51
 * @Description:
 */
@RestController
@Api(tags = "标注文件上传")
public class MarkFileUploadController {

    private static final Logger log = LoggerFactory.getLogger(MarkFileUploadController.class);

    @Value("${upload.formalPath}")
    private String formalPath;

    @Autowired
    private DataSonService dataSonService;

    @Autowired
    private FileService fileService;

    @Autowired
    private MarkInfoMapper markInfoMapper;
    @Autowired
    private FileMapper fileMapper;
    @Autowired
    private DataSonMapper dataSonMapper;

    @PostMapping("/mark/MarkFileUpload")
    public Result MarkFileUpload(@ModelAttribute MarkFileRequest request) throws IOException {
        if (ObjectUtils.isEmpty(request.getFile())) {
            return Result.fail("上传的文件不能为空！");
        }
        if (StringUtils.isEmpty(request.getSonId())) {
            return Result.fail("数据集id不能为空！");
        }
        if (request.getVersion() == null) {
            return Result.fail("版本不能为空！");
        }

        DataSonEntity entity = dataSonService.getOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, request.getSonId())
        );
        if (ObjectUtils.isEmpty(entity)) {
            return Result.fail("数据集不存在！");
        }

        // 组织上传路径
        String filePath = formalPath + entity.getFatherId() + "/" + "v" + entity.getVersion() + "/" + "mark" + "/";
        File fileMkdir = new File(filePath);
        // 判断目录是否存在，不存在，则创建
        if (!fileMkdir.exists()) {
            fileMkdir.mkdirs();
        }

        InputStream in = null;
        FileOutputStream out = null;
        FileEntity fileEntity = new FileEntity();

        String fileName = request.getFile().getOriginalFilename().substring(0, request.getFile().getOriginalFilename().lastIndexOf(".")) + "(mark)" +
                request.getFile().getOriginalFilename().substring(request.getFile().getOriginalFilename().lastIndexOf("."));

        // 执行上传
        try {
            // 文件输入流
            in = request.getFile().getInputStream();
            // 文件输出流
            out = new FileOutputStream(filePath + fileName);
            // 一次读1024个字节
            byte[] buf = new byte[1024];
            int readLen = 0;
            while ((readLen = in.read(buf)) != -1) {
                // 读到数据以后，就写入
                out.write(buf, 0, readLen);
            }


            fileEntity.setFdPath(filePath + fileName);
            fileEntity.setFdName(fileName);
            fileEntity.setFdType(request.getFile().getContentType());
            fileEntity.setFdSuffix(Objects.requireNonNull(request.getFile().getOriginalFilename()).substring(request.getFile().getOriginalFilename().lastIndexOf(".")));
            fileEntity.setFdAccessPath(request.getFile().getOriginalFilename());
            fileEntity.setFdSize(FileFormatSizeUtils.formatSize(request.getFile().getSize()));
            fileEntity.setCreateTime(new Date());


            String shrinkPath = null;
            // 生成缩略图
            if (!fileEntity.getFdSuffix().equals(".json")) {
                shrinkPath = saveShrinkFile(
                        formalPath + entity.getFatherId() + "/" + "v" + entity.getVersion()
                        , filePath + "/" + fileEntity.getFdName());
            }

            fileEntity.setHttpFilePath(shrinkPath);


            boolean result = fileService.save(fileEntity);
            if (!result) {
                throw new RuntimeException("后端异常，标注文件数据新增失败！");
            }


        } catch (Exception e) {
            log.error("-----------------Upload File Fail-----------------");
            throw new RuntimeException(e);
        } finally {
            if (in != null) {
                in.close();
            }
            if (out != null) {
                out.close();
            }
        }
        log.info("-----------------Upload File Successfully-----------------");


        return Result.ok(fileEntity.getId());
    }


    // 删除文件
    @DeleteMapping("/mark/deleteFile")
    public Result deleteFile(@RequestBody FileDeleteRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("参数对象不能为空！");
        }
        DataSonEntity dataSonEntity = dataSonService.getOne(
                new LambdaQueryWrapper<DataSonEntity>()
                        .eq(DataSonEntity::getSonId, request.getSonId())
        );
        if (ObjectUtils.isEmpty(dataSonEntity)) {
            return Result.fail("数据集不存在！");
        }

        for (Integer fileId : request.getFileIds()) {
            List<String> fileIds = Arrays.asList(dataSonEntity.getFileIds().split(","));
            if (!fileIds.contains(fileId.toString())) {
                continue;
            }
            FileEntity fileEntity;
            //---------------------------------- 删除源文件 -------------------------------------
            fileEntity = fileService.getById(fileId);
            File file = new File(fileEntity.getFdPath());
            if (file.exists()) {
                file.delete();
            }

            //---------------------------------- 删除标注文件 -------------------------------------
            MarkInfoEntity markInfoEntity = markInfoMapper.selectOne(
                    new LambdaQueryWrapper<MarkInfoEntity>()
                            .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                            .eq(MarkInfoEntity::getFileId, fileId)
            );
            if (!ObjectUtils.isEmpty(markInfoEntity)) {
                fileEntity = fileService.getById(markInfoEntity.getMarkFileId());
                File file1 = new File(fileEntity.getFdPath());
                if (file1.exists()) {
                    file1.delete();
                }

                //---------------------------------- 删除标注文件正式表数据 -------------------------------------
                if (fileMapper.deleteById(markInfoEntity.getMarkFileId()) < SystemConstant.MAX_SIZE) {
                    return Result.fail("删除标注文件正式表数据失败！");
                }
                //---------------------------------- 删除标注文件信息表数据 -------------------------------------
                if (markInfoMapper.deleteById(markInfoEntity.getId()) < SystemConstant.MAX_SIZE) {
                    return Result.fail("删除标注文件信息表数据失败！");
                }
            }

            //---------------------------------- 修改数据集表数据 -------------------------------------
            List<String> fileList = new ArrayList<String>(fileIds);//转换为ArrayList调用相关的remove方法
            fileList.remove(fileId.toString());
            dataSonEntity.setFileIds(com.qczy.utils.StringUtils.strip(Arrays.toString(fileList.toArray()), "[]").replaceAll("\\s*", ""));
            dataSonMapper.updateById(dataSonEntity);
        }

        // 记录进度总数
        // 计算进度
        if (!ObjectUtils.isEmpty(dataSonEntity)) {
            if (StringUtils.isEmpty(dataSonEntity.getFileIds())) {
                dataSonEntity.setStatus("0% (0/0)");
            } else {
                String[] fileIds = dataSonEntity.getFileIds().split(",");
                Integer count = markInfoMapper.selectCount(
                        new LambdaQueryWrapper<MarkInfoEntity>()
                                .eq(MarkInfoEntity::getSonId, dataSonEntity.getSonId())
                );
                int num = NumberUtil.div(count.toString(), Integer.toString(fileIds.length), 2).multiply(BigDecimal.valueOf(100)).intValue();
                dataSonEntity.setStatus(num + "% " + ("(" + count + "/" + fileIds.length + ")"));
            }
            dataSonMapper.updateById(dataSonEntity);
        }


        return Result.ok(1);
    }


    // 文件缩放
    private String saveShrinkFile(
            String dirPath,
            String newFilePath) {
        try {
            File file = new File(newFilePath);
            if (!file.exists()) { // 判断文件是否穿在
                return null;
            }
            String newPathStr = dirPath + "/" + "shrin" + "/";
            File dirPathStr = new File(newPathStr);
            if (!dirPathStr.exists()) { //判断当前目录是否存在
                dirPathStr.mkdirs();
            }

            // 开始新增缩放
            Thumbnails.of(file).
                    //scalingMode(ScalingMode.BICUBIC).
                            scale(0.1). // 图片缩放80%, 不能和size()一起使用
                    outputQuality(1.0). // 图片质量压缩80%
                    toFile(newPathStr + file.getName());

            return newPathStr + file.getName();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


    // 删除文件
    @DeleteMapping("/mark/deleteResultFile")
    public Result deleteResultFile(@RequestBody FileDeleteRequest request) {
        if (ObjectUtils.isEmpty(request)) {
            return Result.fail("参数对象不能为空！");
        }


        for (Integer fileId : request.getFileIds()) {
            FileEntity fileEntity;
            //---------------------------------- 删除源文件 -------------------------------------
            fileEntity = fileService.getById(fileId);
            if (ObjectUtils.isEmpty(fileEntity)) {
                continue;
            }
            File file = new File(fileEntity.getFdPath());
            if (!file.exists()) {
                break;
            }
            file.delete();
            //---------------------------------- 删除文件表数据 -------------------------------------
            fileMapper.deleteById(fileId);
        }

        return Result.ok(1);
    }


    public static void main(String[] args) {
        new File("H:/qczy/formal/1278031985909956608/v1/source/猫咪1.png").delete();

    }

}

