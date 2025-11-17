package com.qczy.controller.download;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;

import com.qczy.mapper.ModelAssessConfigMapper;
import com.qczy.mapper.ModelAssessTaskMapper;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.ModelAssessConfigEntity;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.request.DownloadRequest;
import com.qczy.service.DataSonService;
import com.qczy.service.ModelAssessService;
import com.qczy.utils.FolderToZipUtil;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.util.ObjectUtils;
import org.springframework.web.bind.annotation.*;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/9/5 10:56
 * @Description:
 */
@RestController
@Api(tags = "文件下载")
public class FileDownloadController {

    private static final Logger log = LoggerFactory.getLogger(FileDownloadController.class);

    @Autowired
    private DataSonService dataSonService;


    @Autowired
    private ResourceLoader resourceLoader;

    @Autowired
    private ModelAssessTaskMapper modelAssessTaskMapper;
    @Autowired
    private ModelAssessConfigMapper modelAssessConfigMapper;


    @Value("${upload.formalPath}")
    private String formalPath;

    @PostMapping("/file/download")
    public void fileDownload(@RequestBody DownloadRequest request, HttpServletResponse response) {
        try {
            if (StringUtils.isBlank(request.getSonId())) {
                throw new RuntimeException("数据集id不能为空！");
            }
            DataSonEntity dataSonEntity = dataSonService.getOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, request.getSonId())
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("数据集对象不存在！");
            }
            // 拼接需要下载的文件路径
            String downloadPath = formalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion();
            File dir = new File(downloadPath);
            log.info("需要下载的文件目录{}", dir.getPath());
            if (!dir.exists()) {
                throw new RuntimeException("文件目录不存在，下载失败！");
            }
            log.info("---------------------------------文件开始下载---------------------------------");


            FolderToZipUtil.zip(String.valueOf(dir), dataSonEntity.getFatherId(), response, request.getType(), request.getAnoType());

        } catch (Exception e) {
            log.error("---------------------------------文件下载失败---------------------------------");
            log.error(e.getMessage());
        }

        log.info("---------------------------------文件下载成功---------------------------------");

    }


    @GetMapping("/download/zip")
    public void zip1(DownloadRequest request, HttpServletResponse response) {
        try {
            if (request == null || request.getId() == null) {
                throw new RuntimeException("数据集id不能为空！");
            }


            ModelAssessConfigEntity modelAssessConfigEntity = modelAssessConfigMapper.selectOne(
                    new LambdaQueryWrapper<ModelAssessConfigEntity>()
                            .eq(ModelAssessConfigEntity::getAssessTaskId, request.getId())
            );
            if (modelAssessConfigEntity == null) {
                return;
            }
            DataSonEntity dataSonEntity = dataSonService.getOne(
                    new LambdaQueryWrapper<DataSonEntity>()
                            .eq(DataSonEntity::getSonId, modelAssessConfigEntity.getSonId())
            );
            if (ObjectUtils.isEmpty(dataSonEntity)) {
                throw new RuntimeException("数据集对象不存在！");
            }
            // 拼接需要下载的文件路径
            String downloadPath = formalPath + dataSonEntity.getFatherId() + "/v" + dataSonEntity.getVersion();
            File dir = new File(downloadPath);
            log.info("需要下载的文件目录{}", dir.getPath());
            if (!dir.exists()) {
                throw new RuntimeException("文件目录不存在，下载失败！");
            }
            log.info("---------------------------------文件开始下载---------------------------------");


            FolderToZipUtil.zip(String.valueOf(dir), dataSonEntity.getFatherId(), response, 3, null);

        } catch (Exception e) {
            log.error("---------------------------------文件下载失败---------------------------------");
            log.error(e.getMessage());
        }

        log.info("---------------------------------文件下载成功---------------------------------");

    }

    @PostMapping("/file/TemDownload")
    public void TemDownload(@RequestParam Integer type, HttpServletResponse response) {
        try {
            String resourcePath = "classpath:/template/";
            String fileName;

            switch (type) {
                case 1:
                    fileName = "标签批量导入模板.xlsx";
                    resourcePath += fileName;
                    break;
                default:
                    response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                    return;
            }

            Resource resource = resourceLoader.getResource(resourcePath);
            if (!resource.exists()) {
                log.error("文件不存在: {}", resourcePath);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }

            // 设置正确的 Content-Type
            response.setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");

            // RFC 5987 标准文件名编码（推荐）
            String encodedFileName = URLEncoder.encode(fileName, String.valueOf(StandardCharsets.UTF_8)).replace("+", "%20");
            response.setHeader("Content-Disposition", "attachment; filename*=UTF-8''" + encodedFileName);

            // 写入文件流
            try (InputStream is = resource.getInputStream();
                 OutputStream os = response.getOutputStream()) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                os.flush();
            }
            log.info("文件下载成功: {}", fileName);
        } catch (Exception e) {
            log.error("文件下载失败", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }


    // 下载模板
    @GetMapping("/api/downloadTemplate")
    public void downloadTemplate(String serverKey, HttpServletResponse response) {
        if (StringUtils.isEmpty(serverKey)) {
            return;
        }

        String fileName = null;
        String filePath = null;

        switch (serverKey) {
            case "modelAlgorithmCode":
                fileName = "codeTemplate.xlsx"; // 模板文件名
                filePath = "templates/" + fileName; // 模板文件在resources下的路径
                break;

            case "testCase":
                fileName = "trainTemplate.xlsx";
                filePath = "templates/" + fileName;
                break;


            default:
                try {
                    response.sendError(HttpServletResponse.SC_NOT_FOUND, "模板文件不存在");
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                return;
        }

        //设置响应头允许跨域访问时暴露Content-Disposition
        response.addHeader("Access-Control-Expose-Headers", "Content-Disposition"); // 关键行
        response.setContentType("application/octet-stream");
        response.setCharacterEncoding("utf-8");
        response.setHeader("Content-Disposition", "attachment;filename=" + fileName);

        // 从classpath读取文件
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(filePath);
             ServletOutputStream os = response.getOutputStream()) {

            if (is == null) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "模板文件不存在");
                return;
            }

            // 复制文件流
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            os.flush();

        } catch (IOException e) {
            // 记录异常日志
            log.error("下载模板文件失败", e);
            try {
                response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "下载文件失败");
            } catch (IOException ex) {
                log.error("设置错误响应失败", ex);
            }
        }
    }

    private static List<File> getAllFolders(File dir) {
        List<File> folders = new ArrayList<>();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    folders.add(file);
                    folders.addAll(getAllFolders(file));
                }
            }
        }
        return folders;
    }

}
