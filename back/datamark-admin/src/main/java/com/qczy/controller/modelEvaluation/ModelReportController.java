package com.qczy.controller.modelEvaluation;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.common.result.Result;
import com.qczy.model.entity.ModelAssessTaskEntity;
import com.qczy.model.entity.ModelBaseEntity;
import com.qczy.model.response.DownloadResponse;
import com.qczy.model.response.ModelAssessResponse;
import com.qczy.model.response.ModelReportResponse;
import com.qczy.service.ModelAssessService;
import com.qczy.service.ModelBaseService;
import com.qczy.utils.StringUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.CollectionUtils;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.List;

@RestController
@Api(tags = "评估报告")
public class ModelReportController {

    @Autowired
    private ModelAssessService modelAssessService;
    @Autowired
    private ModelBaseService modelBaseService;

    /**
     * 测试评估报告-列表
     */
    @GetMapping("/model/report/listPage")
    @ApiOperation("测试评估报告-列表")
    public Result listPage(@RequestParam Integer page,
                           @RequestParam Integer limit) {
        Page<ModelReportResponse> pageParam = new Page<>(page, limit);
        IPage<ModelReportResponse> list = modelAssessService.reportListPage(pageParam);
        return Result.ok(list);
    }


    /**
     * 生成评估报告
     */
    @GetMapping("/api/download")
    @ApiOperation("生成评估报告")
    public void generatePad(DownloadResponse DownloadResponse, HttpServletRequest request, HttpServletResponse response) {
        if (StringUtils.isEmpty(DownloadResponse.getSign())) {
            return;
        }
        switch (DownloadResponse.getSign()) {
            case "task":
                modelAssessService.generateWord(Integer.parseInt(DownloadResponse.getId()), request, response);
                break;
            case "apply":
                // 查看当前的模型id
                List<ModelBaseEntity> list = modelBaseService.list(
                        new LambdaQueryWrapper<ModelBaseEntity>()
                                .eq(ModelBaseEntity::getApplyForNum, DownloadResponse.getId())
                );
                if (CollectionUtils.isEmpty(list)) {
                    return;
                }
                modelAssessService.generateApplyNoWordZip(list.get(0).getId(), request, response);
                break;
        }

    }


}
