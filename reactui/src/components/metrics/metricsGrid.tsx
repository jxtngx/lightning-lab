import { Container, Grid } from "@mui/material";

import { MetricOne } from "./metricOne";
import { MetricTwo } from "./metricTwo";
import { MetricThree } from "./metricThree";
import { MetricFour } from "./metricFour";

export const MetricsGrid = () => (
    <Grid container spacing={3} direction={"row"}>
        <Grid item lg={3} sm={6} xl={3} xs={12}>
            <MetricOne />
        </Grid>
        <Grid item xl={3} lg={3} sm={6} xs={12}>
            <MetricTwo />
        </Grid>
        <Grid item xl={3} lg={3} sm={6} xs={12}>
            <MetricThree />
        </Grid>
        <Grid item xl={3} lg={3} sm={6} xs={12}>
            <MetricFour />
        </Grid>
    </Grid>
);
