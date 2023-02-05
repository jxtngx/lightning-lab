import { Card, CardContent, Grid, Typography } from "@mui/material";

const groundTruth = (props: any) => (
    <Card sx={{ height: "100%" }} {...props}>
        <CardContent>
            <Grid container spacing={3} sx={{ justifyContent: "space-between" }}>
                <Grid item>
                    <Typography color="textPrimary" variant="h4">
                        Metric One
                    </Typography>
                </Grid>
                <Grid item>
                    <Typography
                        color="pass"
                        sx={{
                            mr: 1,
                        }}
                        variant="h4"
                    >
                        .xx%
                    </Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);

const decodedImage = (props: any) => (
    <Card sx={{ height: "100%" }} {...props}>
        <CardContent>
            <Grid container spacing={3} sx={{ justifyContent: "space-between" }}>
                <Grid item>
                    <Typography color="textPrimary" variant="h4">
                        Metric One
                    </Typography>
                </Grid>
                <Grid item>
                    <Typography
                        color="pass"
                        sx={{
                            mr: 1,
                        }}
                        variant="h4"
                    >
                        .xx%
                    </Typography>
                </Grid>
            </Grid>
        </CardContent>
    </Card>
);

export const imageGrid = (props: any) => (
    <></>
)
