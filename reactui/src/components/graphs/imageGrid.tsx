import { Container, Grid, Box, Typography } from "@mui/material";

export const ImageGrid = (props: any) => (
    <Container>
        <Grid container spacing={4}>
            <Grid item>
                <Box
                    sx={{
                        width: 300,
                        height: 300,
                        backgroundColor: "primary.light",
                    }}
                >
                    <Typography>Ground Truth</Typography>
                </Box>
            </Grid>
            <Grid item>
                <Box
                    sx={{
                        width: 300,
                        height: 300,
                        backgroundColor: "primary.light",
                    }}
                >
                    <Typography>Decoded Image</Typography>
                </Box>
            </Grid>
        </Grid>
    </Container>
);
